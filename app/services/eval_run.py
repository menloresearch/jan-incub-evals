from typing import Dict, Type, Any
from ..schemas.eval_run import EvalRunConfig, EvalRunStatus


class EvalRun:
    """
    Plain Python class for runtime evaluation run processing.
    Provides mutable data structures for efficient processing.
    """

    # Grader factory registry - import here to avoid circular imports
    @property
    def GRADER_REGISTRY(self) -> Dict[str, Type[Any]]:
        """Lazy import of grader registry to avoid circular imports."""
        if not hasattr(self, "_grader_registry"):
            from .grader import StringCheckGrader

            self._grader_registry = {
                "string_check": StringCheckGrader,
                # Add other graders here as they're implemented:
                # "label_model": LabelModelGrader,
                # "text_similarity": TextSimilarityGrader,
                # "python": PythonGrader,
                # "score_model": ScoreModelGrader,
            }
        return self._grader_registry

    def __init__(self, config: EvalRunConfig):
        """Initialize from EvalRunConfig with mutable data structures."""
        self.id = config.id
        self.eval_id = config.eval_id
        self.created_at = config.created_at
        self.status = config.status
        self.report_url = config.report_url
        self.metadata = dict(config.metadata)
        self.model = config.model
        self.name = config.name
        self.error = config.error
        self.per_model_usage = config.per_model_usage
        self.per_testing_criteria_results = config.per_testing_criteria_results
        self.result_counts = config.result_counts

        # Convert data_source content to mutable dictionaries
        self.data_source = config.data_source
        self.content_items = []

        # Extract content items as mutable dictionaries
        if hasattr(config.data_source.source, "content"):
            for item in config.data_source.source.content:
                if hasattr(item, "model_dump"):  # Pydantic object
                    self.content_items.append(item.model_dump())
                else:  # Already a dict
                    self.content_items.append(dict(item))

    def add_result_fields(self) -> None:
        """Add result fields (run_id, data_source_idx, grades, etc.) to content items."""
        for idx, item in enumerate(self.content_items):
            item["run_id"] = self.id
            item["data_source_idx"] = idx
            item["grades"] = {}
            item["grader_samples"] = {}
            item["passes"] = {}

    def update_status(self, status: EvalRunStatus) -> None:
        """Update the run status."""
        self.status = status

    def create_grader(self, config) -> Any:
        """Factory method to create grader instances from config."""
        if config.type not in self.GRADER_REGISTRY:
            raise ValueError(f"Unknown grader type: {config.type}")

        grader_class = self.GRADER_REGISTRY[config.type]
        return grader_class(config)

    def to_config(self) -> EvalRunConfig:
        """Convert back to EvalRunConfig for API responses and storage."""
        # Create a copy of data_source and update its content
        data_source_dict = self.data_source.model_dump()
        data_source_dict["source"]["content"] = self.content_items

        return EvalRunConfig(
            id=self.id,
            eval_id=self.eval_id,
            created_at=self.created_at,
            status=self.status,
            report_url=self.report_url,
            metadata=self.metadata,
            model=self.model,
            name=self.name,
            error=self.error,
            per_model_usage=self.per_model_usage,
            per_testing_criteria_results=self.per_testing_criteria_results,
            result_counts=self.result_counts,
            data_source=data_source_dict,
        )
