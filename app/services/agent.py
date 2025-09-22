# This file implements a simple "agent" = OpenAI endpoint + MCP tools.
# Current limitations:
# - Single user turn.
# - Only stdio MCPs are supported
# - No streaming support (only important if we need this for UI)
#
# Run this inside an async function
#
# agent = Agent("http://10.200.108.91:1234/v1")
# async with agent:
#     history, answer = agent.generate_response("What is Menlo Research?")

"""
Now I can compare the parameters:

 CompletionsRunDataSource available parameters:

 - model: str
 - sampling_params: Optional[SamplingParamsSchema] with fields:
   - max_completions_tokens: Optional[int]
   - response_format: Optional[Any]
   - seed: Optional[int]
   - temperature: Optional[float] (default=1)
   - tools: Optional[List[Any]]
   - top_p: Optional[float]

 Agent class parameters:

 - model: str | None (auto-detected if None)
 - sampling_params: dict (default empty)
 - base_url: str (required - OpenAI endpoint URL)
 - api_key: str | None
 - system_prompt: str | None
 - mcp_servers: list (for MCP tool integration)

 Analysis:

 ✅ Available in both: model, sampling_params (compatible - agent expects dict, schema provides structured fields)

 ❌ Missing from CompletionsRunDataSource but needed by Agent:
 - base_url: Required - the OpenAI-compatible endpoint URL
 - api_key: Optional but often needed for authentication
 - system_prompt: Optional but useful for evaluation contexts
 - mcp_servers: Optional but needed if you want tool usage

 The sampling parameters are compatible since Agent expects a dict and you can convert the SamplingParamsSchema to dict.
  However, you need to add at minimum the base_url parameter to use agent.py effectively.
"""

import asyncio
import dataclasses
import json
import logging
import os
import mcp
import openai
import requests
from openai.types.chat import ChatCompletionMessage
from app.utils import substitute_template_variables

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class McpClient:
    """A light wrapper for convenience.

    Instead of having to do

    ```python
    async with mcp.stdio_client(params) as (read, write):
        async with mcp.ClientSession(read, write) as sess:
            ...
    ```

    You can do

    ```python
    async with McpClient(params) as client:
        # client._sess -> session object
        # client.tools -> for `/chat/completions`
        ...
    ```

    It also dynamically supports either MCP stdio or MCP Streamable HTTP.
    To use MCP Streamable HTTP, you need to provide a server_url and headers.

    For Streamable HTTP:
        The server_url is the URL of the MCP server and the headers are the headers to send to the MCP server.
        The headers are optional and will default to an empty dictionary if not provided.
        The server_url is required and will default to None if not provided.
    For MCP stdio:
        The params are the parameters to pass to the MCP stdio server.
    """

    type: str
    command: str | None = None
    args: list[str] | None = None
    server_url: str | None = None
    env: dict | None = None
    headers: dict | None = None
    tools: list = dataclasses.field(default=list)
    allowed_tools: list = dataclasses.field(default_factory=list)

    async def __aenter__(self):
        """Initialize MCP"""
        if self.type == "stdio":
            self.params = mcp.StdioServerParameters(
                command=self.command, args=self.args, env=self.env
            )
            self._stdio_ctx = mcp.stdio_client(self.params)
            read, write = await self._stdio_ctx.__aenter__()
        elif self.type == "http":
            self._http_ctx = mcp.client.streamable_http.streamablehttp_client(
                url=self.server_url,
                headers=self.headers or {},
                sse_read_timeout=300,  # NOTE: 5 minutes seems like a reasonable timeout
            )
            read, write, _ = await self._http_ctx.__aenter__()
        else:
            raise RuntimeError(f"Unsupported MCP type: {self.type}")

        self._sess_ctx = mcp.ClientSession(read, write)
        self._sess = await self._sess_ctx.__aenter__()
        await self._sess.initialize()
        if not isinstance(self.allowed_tools, list):
            self.allowed_tools = [self.allowed_tools]

        mcp_tools = (await self._sess.list_tools()).tools

        # Convert MCP format to OpenAI format (same as McpClient)
        self.tools = [
            dict(
                type="function",
                function=dict(
                    name=mcp_tool.name,
                    description=mcp_tool.description,
                    parameters=mcp_tool.inputSchema,
                ),
            )
            for mcp_tool in mcp_tools
            if not self.allowed_tools or mcp_tool.name in self.allowed_tools
        ]  # Conditional for loop that doesnt filter any tools if allowed_tools is not provided (empty list)

        # Sort dictionary keys (same workaround as McpClient)
        self.tools = json.loads(json.dumps(self.tools, sort_keys=True))

        return self

    async def __aexit__(self, exc_type, exc, tb):
        # cleanup in reverse order
        await self._sess_ctx.__aexit__(exc_type, exc, tb)

        if self.type == "stdio":
            await self._stdio_ctx.__aexit__(exc_type, exc, tb)
        elif self.type == "http":
            await self._http_ctx.__aexit__(exc_type, exc, tb)

        self.tools = []
        del self._sess

    async def call_tool(self, name: str, arguments: dict | None = None):
        return await self._sess.call_tool(name, arguments)


@dataclasses.dataclass
class Agent:
    base_url: str
    api_key: str | None = ""
    model: str | None = None
    template: list | None = None
    mcp_servers: list = dataclasses.field(default_factory=list)
    max_retries: int = 10
    sampling_params: dict = dataclasses.field(default_factory=dict)

    # Internal fields populated from template during __post_init__
    system_prompt: str | None = dataclasses.field(default=None, init=False)
    developer_prompt: str | None = dataclasses.field(default=None, init=False)
    seeded_history: list = dataclasses.field(default_factory=list, init=False)

    def __post_init__(self):
        # Process template if provided
        if self.template is not None:
            self._process_template()

        # replace with env var
        if self.api_key.startswith("env."):
            name = self.api_key.removeprefix("env.")
            self.api_key = os.environ.get(name, "")

        # retry logic is implemented separately in self.make_request()
        self.client = openai.AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key, max_retries=1
        )
        self.tools = None

        # auto-detect model
        if self.model is None:
            headers = dict()
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            resp = requests.get(f"{self.base_url}/models", headers=headers)
            resp.raise_for_status()
            self.model = resp.json()["data"][0]["id"]

        LOGGER.info(f"Using model={self.model} with {self.sampling_params=}")

    def _process_template(self):
        """Process template messages and extract system/developer prompts and seeded history"""
        system_messages = []
        developer_messages = []
        user_assistant_messages = []

        for message in self.template:
            role = message.get("role")
            content = message.get("content")

            if role == "system":
                system_messages.append(content)
            elif role == "developer":
                developer_messages.append(content)
            elif role in ["user", "assistant"]:
                user_assistant_messages.append({"role": role, "content": content})
            else:
                raise ValueError(
                    f"Invalid message role: {role}. Must be one of: user, assistant, system, developer"
                )

        # Validate maximum of 1 system and 1 developer prompt
        if len(system_messages) > 1:
            raise ValueError(
                f"Only one system message allowed, found {len(system_messages)}"
            )
        if len(developer_messages) > 1:
            raise ValueError(
                f"Only one developer message allowed, found {len(developer_messages)}"
            )

        # Set the prompts
        self.system_prompt = system_messages[0] if system_messages else None
        self.developer_prompt = developer_messages[0] if developer_messages else None
        self.seeded_history = user_assistant_messages

    async def __aenter__(self):
        """Initialize MCPs"""
        self._mcp_contexts: list = []  # store context managers for proper cleanup
        self._mcps: list[McpClient] = []  # for cleanup later
        self._tool_to_mcp: dict[str, McpClient] = dict()  # lookup client from tool name
        self.tools = []

        for args in self.mcp_servers:
            # Store the context manager and enter it
            client = McpClient(**args)
            self._mcp_contexts.append(client)
            await client.__aenter__()

            for tool in client.tools:
                tool_name = tool["function"]["name"]
                assert tool_name not in self._tool_to_mcp
                self._tool_to_mcp[tool_name] = client
                self.tools.append(tool)

        LOGGER.info(f"tools={self.tools}")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # cleanup in reverse order, ensuring all clients are cleaned up even if some fail
        # Try and except here is to ensure that if one client fails to cleanup, the other clients are still cleaned up
        # This is a workaround for https://github.com/modelcontextprotocol/python-sdk/issues/922
        for client in reversed(self._mcp_contexts):
            try:
                await client.__aexit__(exc_type, exc, tb)
            except Exception as e:
                LOGGER.exception(f"Failed to cleanup mcp client: {e}")
                raise e
        self.tools = None

        del self._tool_to_mcp
        del self._mcps
        del self._mcp_contexts

    async def generate_response(
        self,
        prompt: str,
        tool_use: bool = True,
        history: list | None = None,
        item_data: dict = None,
        sample_data: dict = None,
    ):
        if self.mcp_servers:
            assert self.tools is not None, "MCP servers found, but not initialized."

        # Build messages in correct order: developer -> system -> seeded_history -> history -> user prompt
        messages = []

        # Add developer prompt first (highest precedence)
        if self.developer_prompt is not None:
            developer_content = substitute_template_variables(
                self.developer_prompt, item_data, sample_data
            )
            messages.append(dict(role="developer", content=developer_content))

        # Add system prompt second
        if self.system_prompt is not None:
            system_content = substitute_template_variables(
                self.system_prompt, item_data, sample_data
            )
            messages.append(dict(role="system", content=system_content))

        # Add seeded history (user/assistant messages from template)
        for message in self.seeded_history:
            substituted_message = dict(message)  # Create a copy
            substituted_message["content"] = substitute_template_variables(
                message["content"], item_data, sample_data
            )
            messages.append(substituted_message)

        # Add runtime history if provided
        if history is not None:
            messages.extend(history)

        # Add the new user prompt (also apply substitution)
        user_content = substitute_template_variables(prompt, item_data, sample_data)
        messages.append(dict(role="user", content=user_content))

        final_response = None

        # tool call loop
        # https://platform.openai.com/docs/guides/function-calling?api-mode=chat
        while True:
            message = await self.make_request(messages, tool_use=tool_use)
            messages.append(message.to_dict())

            # final message is received when the model doesn't want to make tool calls
            if not message.tool_calls:
                final_response = message.content
                break

            # NOTE: if we fail to call tool, we simply append an error message. the model can decide to
            # retry by itself.
            # a possible alternative is to resend the previous request until a valid tool call is received,
            # which is not implemented here.
            for tool_call in message.tool_calls:
                try:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    mcp_result = await self._tool_to_mcp[name].call_tool(name, args)

                    # NOTE: content might be an empty list
                    content = [
                        dict(type="text", text=x.text)
                        for x in mcp_result.content
                        if x.type == "text"
                    ]

                except Exception as e:
                    # error can happen when
                    # 1. model outputs wrong format e.g. wrong tool name/arguments, missing required arguments
                    # 2. downstream error in MCP server

                    LOGGER.exception(f"Failed to call tool {name=}, {args=}. {prompt=}")

                    error_type = type(e).__name__
                    error_details = str(e)
                    content = f"Error calling tool '{name}': {error_type}. Details: {error_details}"

                messages.append(
                    dict(role="tool", tool_call_id=tool_call.id, content=content)
                )

        if final_response is None:
            final_response = ""

        return messages, final_response

    async def make_request(self, messages: list[dict], tool_use: bool = True):
        """Make `/chat/completions` request with retry loop"""
        # not sure why official OpenAI SDK doesn't have exponential backoff feature
        # https://platform.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff

        for retry in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools
                    if (self.tools is not None and tool_use)
                    else openai.NOT_GIVEN,
                    **self.sampling_params,
                )
                return response.choices[0].message

            # follow OpenAI
            # https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/sampler/chat_completion_sampler.py#L81C13-L82
            except openai.BadRequestError:
                LOGGER.exception("Bad request")
                return ChatCompletionMessage(
                    role="assistant", content="No response (bad request)"
                )

            except Exception:
                LOGGER.exception(
                    f"Request failed after {retry + 1} retries. Retrying..."
                )
                exception_backoff = 2**retry  # expontial back off
                await asyncio.sleep(exception_backoff)

        raise RuntimeError(f"max_retries={self.max_retries} exceeded")
