import logging
import re

LOGGER = logging.getLogger(__name__)


def substitute_template_variables(
    text: str, item_data: dict = None, sample_data: dict = None
) -> str:
    """
    Replace {{item.field}} and {{sample.field}} variables in text with actual values.

    Args:
        text: String that may contain template variables
        item_data: Dictionary containing item namespace data
        sample_data: Dictionary containing sample namespace data

    Returns:
        str: String with variables replaced
    """
    if not isinstance(text, str):
        return text

    # Pattern to match {{item.field_name}} and {{sample.field_name}}
    pattern = r"\{\{(item|sample)\.([^}]+)\}\}"

    def replace_variable(match):
        namespace = match.group(1)
        field_name = match.group(2)

        if namespace == "item" and item_data and field_name in item_data:
            return str(item_data[field_name])
        elif namespace == "sample" and sample_data and field_name in sample_data:
            return str(sample_data[field_name])
        else:
            LOGGER.warning(
                f"Template variable {{{{{namespace}.{field_name}}}}} not found in {namespace} data"
            )
            return match.group(0)  # Return original if not found

    return re.sub(pattern, replace_variable, text)
