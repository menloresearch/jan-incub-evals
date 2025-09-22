"""
LLM Inference Schemas

This module contains all schemas related to LLM model inference settings:
- SamplingParamsSchema: Model sampling parameters (temperature, top_p, etc.)
- InputMessages: Message templates and structures for model input
"""

from pydantic import BaseModel, Field, validator
from typing import Union, Literal, List, Any, Optional, Dict
import re


class JsonSchemaConfig(BaseModel):
    """Configuration for JSON Schema response format"""

    name: str = Field(
        ...,
        description="The name of the response format. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.",
        max_length=64,
    )
    description: Optional[str] = Field(
        None,
        description="A description of what the response format is for, used by the model to determine how to respond in the format.",
    )
    json_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="The schema for the response format, described as a JSON Schema object.",
    )
    strict: Optional[bool] = Field(
        False,
        description="Whether to enable strict schema adherence when generating the output. If set to true, the model will always follow the exact schema defined in the schema field.",
    )

    @validator("name")
    def validate_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Name must contain only a-z, A-Z, 0-9, underscores, and dashes"
            )
        return v


class FunctionTool(BaseModel):
    """Function definition for tool calling"""

    name: str = Field(
        ...,
        description="The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.",
        max_length=64,
    )
    description: Optional[str] = Field(
        None,
        description="A description of what the function does, used by the model to choose when and how to call the function.",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="The parameters the functions accepts, described as a JSON Schema object. Omitting parameters defines a function with an empty parameter list.",
    )
    strict: Optional[bool] = Field(
        False,
        description="Whether to enable strict schema adherence when generating the function call. If set to true, the model will follow the exact schema defined in the parameters field.",
    )

    @validator("name")
    def validate_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Name must contain only a-z, A-Z, 0-9, underscores, and dashes"
            )
        return v


class Tool(BaseModel):
    """Tool specification for model function calling"""

    type: Literal["function"] = "function"
    function: FunctionTool


class ResponseFormatSchema(BaseModel):
    type: str


class ResponseFormatText(ResponseFormatSchema):
    type: Literal["text"] = "text"


class ResponseFormatJSONSchema(ResponseFormatSchema):
    type: Literal["json_schema"] = "json_schema"
    json_schema: JsonSchemaConfig


class ResponseFormatJSONObject(ResponseFormatSchema):
    type: Literal["json_object"] = "json_object"


ResponseFormatSchemaUnion = Union[
    ResponseFormatText, ResponseFormatJSONSchema, ResponseFormatJSONObject
]


class SamplingParamsSchema(BaseModel):
    # SCHEMA CHANGE: Changed from max_completions_tokens to max_tokens
    # to align with OpenAI API parameter names and eliminate parameter mapping
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens in the generated output.",
    )
    response_format: Optional[ResponseFormatSchemaUnion] = Field(
        default=ResponseFormatText(),
        discriminator="type",
        description='An object specifying the format that the model must output. Setting to { "type": "json_schema", "json_schema": {...} } enables Structured Outputs which ensures the model will match your supplied JSON schema. Learn more in the Structured Outputs guide. Setting to { "type": "json_object" } enables the older JSON mode, which ensures the message the model generates is valid JSON. Using json_schema is preferred for models that support it.',
    )
    seed: Optional[int] = Field(
        default=42,
        description="A seed value to initialize the randomness, during sampling.",
    )
    temperature: Optional[float] = Field(
        default=1,
        description="A higher temperature increases randomness in the outputs.",
    )
    tools: Optional[List[Tool]] = Field(
        default=None,
        description="A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. A max of 128 functions are supported.",
        max_items=128,
    )
    top_p: Optional[float] = Field(
        default=None,
        description="An alternative to temperature for nucleus sampling; 1.0 includes all tokens.",
    )


# Base class for input messages
class InputMessagesSchema(BaseModel):
    type: str


class TemplateInputMessagesTemplateSchema(BaseModel):
    content: str = Field(
        description="A message input to the model with a role indicating instruction following hierarchy. Instructions given with the developer or system role take precedence over instructions given with the user role. Messages with the assistant role are presumed to have been generated by the model in previous interactions."
    )
    role: Literal["user", "assistant", "system", "developer"] = Field(
        description="The role of the message input. One of user, assistant, system, or developer."
    )
    type: Optional[Literal["message"]] = Field(
        default="message", description="Optional since it's always 'message'"
    )


class TemplateInputMessages(InputMessagesSchema):
    type: Literal["template"] = "template"
    template: List[TemplateInputMessagesTemplateSchema]


class ItemReferenceInputMessages(InputMessagesSchema):
    type: Literal["item_reference"] = "item_reference"
    item_reference: str = Field(
        description="A reference to a variable in the item namespace. Ie, 'item.input_trajectory' "
    )


# Union type for input messages
InputMessagesSchemaUnion = Union[TemplateInputMessages, ItemReferenceInputMessages]
