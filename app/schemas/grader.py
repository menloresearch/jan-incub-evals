from pydantic import BaseModel, Field
from typing import Literal, List, Union
from abc import ABC


# Testing criteria structure
class grader_config(BaseModel, ABC):
    """Individual testing criterion for evaluation"""

    type: str
    name: str


# Message structure for graders
class SimpleInputMessage(BaseModel):
    content: str
    role: Literal["system", "assistant", "user"]


class LabelModelGraderConfig(grader_config):
    """A LabelModelGrader object which uses a model to assign labels to each item in the evaluation."""

    type: Literal["label_model"] = "label_model"
    model: str
    input: List[SimpleInputMessage]
    passing_labels: List[str]
    labels: List[str]
    name: str


class StringCheckGraderConfig(grader_config):
    input: str = Field(description="The input text. This may include template strings.")
    type: Literal["string_check"] = "string_check"
    name: str = Field(description="The name of the grader")
    operation: Literal["eq", "ne", "like", "ilike"]
    reference: str = Field(
        description="The reference text. This may include template strings."
    )


class TextSimilarityGraderConfig(grader_config):
    type: Literal["text_similarity"] = (
        "text_similarity"  # there isn't a listed default in openai's docs. but that might be a documentation oversight.
    )


class PythonGraderConfig(grader_config):
    type: Literal["python"] = "python"


class ScoreModelGraderConfig(grader_config):
    type: Literal["score_model"] = "score_model"


# Union type for all graders
GraderConfigUnion = Union[
    LabelModelGraderConfig,
    StringCheckGraderConfig,
    TextSimilarityGraderConfig,
    PythonGraderConfig,
    ScoreModelGraderConfig,
]
