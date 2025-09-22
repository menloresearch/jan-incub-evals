import logging
from datetime import datetime
from typing import List, Tuple
from app.schemas.eval import Eval
from app.schemas.eval_run import EvalRunStatus
from .eval_run import EvalRun
from app.schemas.datasets import CompletionsRunDataSource, ResultContentItemSchema
from .agent import Agent

logger = logging.getLogger(__name__)


class EvalProcessor:
    """Background processor for evaluation runs"""

    def __init__(self, eval_service):
        self.eval_service = eval_service

    async def process_eval_run(self, run_id: str) -> None:
        """
        Process an evaluation run in the background
        """
        try:
            # Update status to running
            self.eval_service.update_run_status(run_id, EvalRunStatus.RUNNING)
            logger.info(f"Started processing eval run {run_id}")

            # Get the run and eval data
            eval_run = self.eval_service.get_eval_run(run_id)
            eval_config = self.eval_service.get_eval(eval_run.eval_id)

            # 1. Validate data source against eval's schema
            is_valid, validation_errors = self.validate_data_source_schema(
                eval_config, eval_run
            )
            if not is_valid:
                logger.error(
                    f"Data source validation failed for eval run {run_id}: {validation_errors}"
                )
                self.eval_service.update_run_status(run_id, EvalRunStatus.FAILED)
                return

            logger.info(f"Data source validation passed for eval run {run_id}")

            # 2. Add result fields to content items (run_id, data_source_idx, etc.)
            eval_run.add_result_fields()
            logger.info(
                f"Added result fields to {len(eval_run.content_items)} content items"
            )

            # 3. Generate completions if necessary
            if isinstance(eval_run.data_source, CompletionsRunDataSource):
                await self._generate_completions(eval_run, eval_config)

            # 4. Run all graders on the content items (modifies in place)
            self.run_graders(eval_run, eval_config)

            # TODO: Implement remaining evaluation run logic here
            # This is where you would:
            # 5. Compile results and produce eval_run_output object

            # 6. Update status to completed
            self.eval_service.update_run_status(run_id, EvalRunStatus.COMPLETED)
            logger.info(f"Completed processing eval run {run_id}")

        except Exception as e:
            logger.error(f"Error processing eval run {run_id}: {str(e)}")
            self.eval_service.update_run_status(run_id, EvalRunStatus.FAILED)

    def validate_data_source_schema(
        self, eval_config: Eval, eval_run: EvalRun
    ) -> Tuple[bool, List[str]]:
        """
        Fast validation that data source content items have all required schema fields

        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        try:
            # Get required fields from eval's item schema (should be empty list if none)
            required_fields = (
                set(eval_config.data_source_config.item_schema.required)
                if eval_config.data_source_config.type == "custom"
                else set()
            )

            if not required_fields:
                return True, []  # No required fields to validate

            # Get content items from the EvalRun object
            # Note: At validation time, content_items are still in their original form
            content_items = eval_run.data_source.source.content

            # Fast validation using set operations
            for i, content in enumerate(content_items):
                # content is a ContentItemSchema dictionary
                if "item" not in content:
                    return False, [
                        f"Content {i}: missing 'item' key. Full content: {content}"
                    ]

                item_fields = set(content["item"].keys())
                missing_fields = required_fields - item_fields

                if missing_fields:
                    return False, [
                        f"Content {i}: missing {sorted(missing_fields)}. Full item: {content['item']}"
                    ]

            return True, []

        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    def run_graders(self, eval_run: EvalRun, eval_config: Eval) -> None:
        """
        Run all graders on the content items, processing grader by grader with lazy initialization.
        Modifies the content_items dictionaries in place.

        Args:
            eval_run: EvalRun object containing content items and grader factory methods
            eval_config: Eval configuration containing testing criteria
        """
        grader_count = 0
        content_items = eval_run.content_items

        print(f"DEBUG: Processing {len(eval_config.testing_criteria)} testing criteria")
        for i, criteria in enumerate(eval_config.testing_criteria):
            print(f"DEBUG: Criteria {i}: {criteria.name} - {criteria.type}")

        for criteria in eval_config.testing_criteria:
            # Create grader on-demand for this criteria using EvalRun's factory
            grader = eval_run.create_grader(criteria)
            grader_count += 1
            print(f"DEBUG: Created grader with name: {grader.name}")

            logger.info(f"Running grader: {grader.name}")

            for item in content_items:
                try:
                    # Create a temporary ResultContentItemSchema for the grader
                    temp_result_item = ResultContentItemSchema(**item)
                    graded_result = grader.score(temp_result_item)

                    # Update the original dictionary with grader results
                    item["grades"][grader.name] = graded_result.grades[grader.name]
                    item["grader_samples"][grader.name] = graded_result.grader_samples[
                        grader.name
                    ]
                    item["passes"][grader.name] = graded_result.passes[grader.name]

                    logger.debug(
                        f"Grader {grader.name} processed item {item['data_source_idx']}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error running grader {grader.name} on item {item['data_source_idx']}: {str(e)}"
                    )
                    # Continue with other items even if one fails

            logger.info(f"Completed grader {grader.name} on {len(content_items)} items")

        logger.info(
            f"Completed all {grader_count} graders on {len(content_items)} items"
        )

    def create_agent(
        self,
        model: str,
        sampling_params,
        input_messages=None,
        base_url: str = None,
        api_key: str = None,
    ) -> Agent:
        """
        Create an Agent instance with the provided parameters.

        Args:
            model: The model name from CompletionsRunDataSource
            sampling_params: SamplingParamsSchema instance
            input_messages: InputMessagesSchemaUnion instance for extracting system prompt
            base_url: Base URL for the LLM endpoint (defaults to eval_service.llm_base_url)
            api_key: API key for the LLM endpoint (defaults to eval_service.llm_api_key)

        Returns:
            Agent: Configured Agent instance
        """
        # Use eval service defaults if not provided
        if base_url is None:
            base_url = self.eval_service.llm_base_url
        if api_key is None:
            api_key = self.eval_service.llm_api_key

        # Pass template directly to Agent if available
        template = None
        if (
            input_messages
            and hasattr(input_messages, "type")
            and input_messages.type == "template"
        ):
            # Convert Pydantic template objects to dictionaries
            template = [msg.model_dump() for msg in input_messages.template]
        elif (
            input_messages
            and hasattr(input_messages, "type")
            and input_messages.type == "item_reference"
        ):
            template = [{"role": "user", "content": input_messages.item_reference}]

        # Convert sampling params to dict and extract mcp_servers from tools
        sampling_params_dict = {}
        mcp_servers = []

        if sampling_params:
            sampling_params_dict = sampling_params.model_dump(exclude_none=True)

            # Extract tools and send to mcp_servers
            mcp_servers = sampling_params_dict.pop("tools", [])

        return Agent(
            base_url=base_url,
            api_key=api_key,
            model=model,
            template=template,
            sampling_params=sampling_params_dict,
            mcp_servers=mcp_servers,
        )

    async def _generate_completions(self, eval_run, eval_config):
        """Generate completions using the Agent with robust completion tracking"""
        try:
            data_source = eval_run.data_source

            # Create Agent using our new create_agent function
            agent = self.create_agent(
                model=data_source.model,
                sampling_params=data_source.sampling_params,
                input_messages=data_source.input_messages,
                base_url=self.eval_service.llm_base_url,
                api_key=self.eval_service.llm_api_key,
            )

            logger.info(f"Created agent with model={data_source.model}")

            # Get content items to process from EvalRun
            content_items = eval_run.content_items
            total_items = len(content_items)

            # Track progress
            completed_count = 0
            failed_count = 0
            skipped_count = 0

            async with agent:
                for i, content_item in enumerate(content_items):
                    try:
                        # Check if this item already has a sample (resume capability)
                        if "sample" in content_item and content_item["sample"]:
                            logger.info(
                                f"Item {i + 1}/{total_items}: Skipping - already has sample data"
                            )
                            skipped_count += 1
                            continue

                        # Extract both item and sample data from content item
                        item_data = content_item.get("item", {})
                        sample_data = content_item.get("sample", {})

                        if not item_data:
                            logger.warning(
                                f"Item {i + 1}/{total_items}: No 'item' data found, skipping"
                            )
                            failed_count += 1
                            continue

                        logger.info(
                            f"Item {i + 1}/{total_items}: Generating completion..."
                        )

                        # Generate response using the agent with template substitution
                        # Pass empty prompt since all templating is handled by the agent
                        history, response = await agent.generate_response(
                            "", item_data=item_data, sample_data=sample_data
                        )

                        # Store the response in the sample field
                        content_item["sample"] = {
                            "output_text": response,
                            "completed_at": int(datetime.now().timestamp()),
                        }

                        # Note: Changes are automatically persisted since we're modifying the stored EvalRun object

                        completed_count += 1
                        logger.info(
                            f"Item {i + 1}/{total_items}: Completion generated and saved"
                        )

                    except Exception as item_error:
                        logger.error(
                            f"Item {i + 1}/{total_items}: Failed to generate completion: {str(item_error)}"
                        )
                        failed_count += 1

                        # Mark item as failed but continue with others
                        content_item["sample"] = {
                            "error": str(item_error),
                            "failed_at": int(datetime.now().timestamp()),
                        }

                        # Note: Changes are automatically persisted since we're modifying the stored EvalRun object

            logger.info(
                f"Completion generation finished: {completed_count} completed, {skipped_count} skipped, {failed_count} failed"
            )

            # If all items failed, consider the whole process failed
            if failed_count > 0 and completed_count == 0 and skipped_count == 0:
                raise Exception(
                    f"All {failed_count} items failed to generate completions"
                )

        except Exception as e:
            logger.error(f"Error generating completions: {str(e)}")
            raise
