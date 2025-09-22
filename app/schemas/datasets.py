"""
Dataset Schemas

This module contains schemas related to data sources and dataset configurations:
- EvalSource: Source implementations (file_content, file_id, responses)
- DataSource: Run data source implementations (jsonl, completions, responses)

"""

from pydantic import BaseModel, Field
from typing import Union, Literal, Dict, List, Any, Optional
from typing_extensions import TypedDict, NotRequired
from abc import ABC

# Import LLM inference schemas from separate module
from .llm_inference import SamplingParamsSchema, InputMessagesSchemaUnion


# ContentItem schema for grader input
ContentItemSchema = TypedDict(
    "ContentItemSchema",
    {
        "item": Dict[str, Any],
        "sample": NotRequired[Dict[str, Any]],
        "run_id": NotRequired[str],
        "data_source_idx": NotRequired[int],
        "grades": NotRequired[Dict[str, int]],
        "grader_samples": NotRequired[Dict[str, Any]],
        "passes": NotRequired[Dict[str, bool]],
    },
)


# Result ContentItem schema with additional fields populated during processing
class ResultContentItemSchema(BaseModel):
    item: Dict[str, Any]
    sample: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None
    data_source_idx: Optional[int] = None
    grades: Optional[Dict[str, int]] = None
    grader_samples: Optional[Dict[str, Any]] = None
    passes: Optional[Dict[str, bool]] = None


# Base Class
class eval_source(
    BaseModel, ABC
):  # this is "source" in openai docs, but the naming is weird so i called it eval_source
    type: str


# specific implementation
class EvalJsonlFileContentSource(eval_source):
    """
    in the content field, the typed dict is important because these are accessed by the various graders and completions models using this format
    {{item.<key of dictionary>}}
    {{sample.<key of dictionary>}}
    """

    type: Literal["file_content"] = "file_content"
    content: List[ContentItemSchema]


class EvalJsonlFieldSource(
    eval_source
):  # this one is listed as "EvalJsonlFileldSource" which has a typo in the OpenAI docs
    type: Literal["file_id"] = "file_id"
    field_id: str = Field(description="The identifier of the file.")


class EvalResponseSource(eval_source):
    """
    A EvalResponsesSource object describing a run data source configuration.
    """

    type: Literal["responses"] = "responses"


# Union type with discriminator
EvalSourceUnion = Union[
    EvalJsonlFileContentSource, EvalJsonlFieldSource, EvalResponseSource
]


# Base class
class data_source(BaseModel, ABC):
    type: str
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Set of 16 key-value pairs for additional information",
    )


# Specific implementations
class JsonlRunDataSource(data_source):
    source: EvalSourceUnion = Field(discriminator="type")
    type: Literal["jsonl"] = "jsonl"
    include_sample_schema: bool = False


class CompletionsRunDataSource(data_source):
    source: EvalSourceUnion = Field(discriminator="type")
    type: Literal["completions"] = "completions"
    input_messages: Optional[InputMessagesSchemaUnion] = Field(
        discriminator="type",
        description="Used when sampling from a model. Dictates the structure of the messages passed into the model. Can either be a reference to a prebuilt trajectory (ie, item.input_trajectory), or a template with variable references to the item namespace.",
    )
    model: str
    sampling_params: Optional[SamplingParamsSchema]


class ResponsesRunDataSource(data_source):
    source: EvalSourceUnion = Field(discriminator="type")
    type: Literal["responses"] = "responses"
    input_messages: dict
    model: str
    sampling_params: Optional[SamplingParamsSchema]


DataSourceUnion = Union[
    JsonlRunDataSource, CompletionsRunDataSource, ResponsesRunDataSource
]
