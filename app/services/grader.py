from abc import ABC, abstractmethod
import re
import uuid
from ..schemas.datasets import ResultContentItemSchema
from ..schemas.grader import StringCheckGraderConfig
from ..utils import substitute_template_variables


class Grader(ABC):
    """Base class for all grader implementations that handle evaluation scoring."""

    def __init__(self, type: str, name: str):
        """
        Initialize the grader with type and name.

        Args:
            type: The type of grader (e.g., "label_model", "string_check")
            name: The name of this grader instance
        """
        self.type = type
        self.name = f"{name}_{uuid.uuid4().hex[:8]}"

    @abstractmethod
    def score(self, content_item: ResultContentItemSchema) -> ResultContentItemSchema:
        """
        Score an evaluation item based on the content and return updated ResultContentItemSchema.

        Args:
            content_item: ResultContentItemSchema containing "item" and optionally "sample" data

        Returns:
            ResultContentItemSchema with updated grades, grader_samples, and passes:
            - grades: Dict where key is self.name and value is the numerical score
            - passes: Dict where key is self.name and value is boolean pass/fail result
            - grader_samples: Dict where key is self.name and value is grader-specific metadata
              (e.g., for StringCheckGrader: store dictionary of grader attributes like input, operation, reference)
        """
        pass


class StringCheckGrader(Grader):
    """Grader that performs string comparison operations."""

    def __init__(self, config: StringCheckGraderConfig):
        super().__init__(config.type, config.name)
        self.input = config.input
        self.operation = config.operation
        self.reference = config.reference

    def score(self, content_item: ResultContentItemSchema) -> ResultContentItemSchema:
        """
        Score based on string comparison operation.

        Args:
            content_item: ResultContentItemSchema containing evaluation data

        Returns:
            ResultContentItemSchema with updated grades, grader_samples, and passes
        """
        # Resolve template variables using the utility function
        input_text = substitute_template_variables(
            self.input,
            item_data=content_item.item if hasattr(content_item, "item") else None,
            sample_data=content_item.sample
            if hasattr(content_item, "sample")
            else None,
        )
        reference_text = substitute_template_variables(
            self.reference,
            item_data=content_item.item if hasattr(content_item, "item") else None,
            sample_data=content_item.sample
            if hasattr(content_item, "sample")
            else None,
        )

        # Strip whitespace to handle whitespace issues
        input_text = input_text.strip() if input_text else ""
        reference_text = reference_text.strip() if reference_text else ""

        # Perform the comparison based on operation
        passed = self._perform_comparison(input_text, reference_text, self.operation)
        score = 1 if passed else 0

        # Update existing grades, grader_samples, and passes or create new ones
        updated_grades = content_item.grades.copy() if content_item.grades else {}
        updated_grader_samples = (
            content_item.grader_samples.copy() if content_item.grader_samples else {}
        )
        updated_passes = content_item.passes.copy() if content_item.passes else {}

        # Add this grader's results
        updated_grades[self.name] = score
        updated_passes[self.name] = passed
        updated_grader_samples[self.name] = {
            "input": self.input,
            "operation": self.operation,
            "reference": self.reference,
            "input_text": input_text,
            "reference_text": reference_text,
        }

        # Return updated ResultContentItemSchema
        return ResultContentItemSchema(
            item=content_item.item,
            sample=content_item.sample,
            run_id=content_item.run_id,
            data_source_idx=content_item.data_source_idx,
            grades=updated_grades,
            grader_samples=updated_grader_samples,
            passes=updated_passes,
        )

    def _perform_comparison(
        self, input_text: str, reference_text: str, operation: str
    ) -> bool:
        """Perform the string comparison based on operation."""
        if operation == "eq":
            return input_text == reference_text
        elif operation == "ne":
            return input_text != reference_text
        elif operation == "like":
            # Convert SQL LIKE pattern to regex
            pattern = reference_text.replace("%", ".*").replace("_", ".")
            return bool(re.match(f"^{pattern}$", input_text))
        elif operation == "ilike":
            # Case-insensitive LIKE
            pattern = reference_text.replace("%", ".*").replace("_", ".")
            return bool(re.match(f"^{pattern}$", input_text, re.IGNORECASE))
        else:
            raise ValueError(f"Unsupported operation: {operation}")
