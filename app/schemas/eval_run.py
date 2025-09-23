from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Optional, TypedDict
from enum import Enum
from .datasets import DataSourceUnion


class EvalRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Schema validation class
class ErrorSchema(BaseModel):
    code: str = Field(description="The error code.")
    message: str = Field(description="The error message.")


class TestingCriteriaResultsSchema(BaseModel):
    failed: int = Field(description="Number of tests failed for this criteria.")
    passed: int = Field(description="Number of tests passed for this criteria.")
    testing_criteria: str = Field(description="A description of the testing criteria.")


class ResultCountsSchema(BaseModel):
    errored: int = Field(
        description="Number of output items that resulted in an error."
    )
    failed: int = Field(
        description="Number of output items that failed to pass the evaluation."
    )
    passed: int = Field(
        description="Number of output items that passed the evaluation."
    )
    total: int = Field(description="Total number of executed output items.")


class EvalRunConfig(BaseModel):
    """
    Configuration for creating an evaluation run.
    Contains only the fields that can be provided by the user via API.
    """

    data_source: DataSourceUnion = Field(
        discriminator="type",
        description="Information about the run's data source.",
    )
    error: Optional[ErrorSchema] = Field(
        default=None,
        description="An object representing an error response from the Eval API.",
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Set of 16 key-value pairs for additional information",
    )
    model: Optional[str] = Field(
        default=None, description="The model that is evaluated, if applicable."
    )
    name: Optional[str] = Field(default=None, description="The name of the run.")
    per_model_usage: Optional[Dict] = Field(
        default=None,
        description="Usage statistics per model used in the evaluation run.",
    )
    per_testing_criteria_results: Optional[List[TestingCriteriaResultsSchema]] = Field(
        default=None,
        description="Results per testing criteria applied during the evaluation run.",
    )
    result_counts: Optional[ResultCountsSchema] = Field(
        default=None,
        description="Counters summarizing the outcomes of the evaluation run.",
    )


class EvalRunResponse(BaseModel):
    """
    Pydantic model for API responses containing EvalRun data.
    Used for validation, serialization, and API contracts.
    """

    created_at: int = Field(
        description="The Unix timestamp (in seconds) for when the eval was created."
    )
    data_source: DataSourceUnion = Field(
        discriminator="type",
        description="Information about the run's data source.",
    )
    error: Optional[ErrorSchema] = Field(
        default=None,
        description="An object representing an error response from the Eval API.",
    )
    eval_id: str = Field(
        description="The ID of the evaluation that this run belongs to."
    )
    id: str = Field(description="Unique identifier for the evaluation run.")
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Set of 16 key-value pairs for additional information",
    )
    model: Optional[str] = Field(
        default=None, description="The model that is evaluated, if applicable."
    )
    name: Optional[str] = Field(default=None, description="The name of the run.")
    object: Literal["eval.run"] = Field(
        default="eval.run", description="The type of the object. Always 'eval.run'."
    )
    per_model_usage: Optional[Dict] = Field(
        default=None,
        description="Usage statistics per model used in the evaluation run.",
    )
    per_testing_criteria_results: Optional[List[TestingCriteriaResultsSchema]] = Field(
        default=None,
        description="Results per testing criteria applied during the evaluation run.",
    )
    report_url: str = Field(
        description="URL to view the detailed report of the evaluation run."
    )
    result_counts: Optional[ResultCountsSchema] = Field(
        default=None,
        description="Counters summarizing the outcomes of the evaluation run.",
    )
    status: EvalRunStatus = Field(description="The status of the evaluation run.")


class MessageDict(TypedDict):
    role: Literal["user", "assistant", "system", "developer"] = Field(
        description="The role of the message input. One of user, assistant, system, or developer."
    )
    content: str = Field(
        description="A message input to the model with a role indicating instruction following hierarchy. Instructions given with the developer or system role take precedence over instructions given with the user role. Messages with the assistant role are presumed to have been generated by the model in previous interactions."
    )


class ErrorDict(TypedDict):
    code: str
    message: str


class UsageDict(TypedDict):
    cached_tokens: int
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class EvalRunOutputItemSampleSchema(BaseModel):
    error: Optional[ErrorDict] = Field(
        default=None,
        description="An object representing an error response from the Eval API.",
    )
    finish_reason: str = Field(
        description="The reason why the sample generation was finished."
    )
    input: List[MessageDict] = Field(description="An array of input messages.")
    max_completion_tokens: int = Field(
        description="The maximum number of tokens allowed for completion."
    )
    model: str = Field(description="The model used for generating the sample.")
    output: List[MessageDict] = Field(description="An array of output messages.")
    seed: int = Field(description="The seed used for generating the sample.")
    temperature: float = Field(description="The sampling temperature used.")
    top_p: float = Field(description="The top_p value used for sampling.")
    usage: UsageDict = Field(description="Token usage details for the sample.")


class EvalRunOutputItem(BaseModel):
    created_at: int = Field(
        description="Unix timestamp (in seconds) when the evaluation run was created."
    )
    datasource_item: Dict = Field(description="Details of the input data source item.")
    datasource_item_id: int = Field(
        description="The identifier for the data source item."
    )
    eval_id: str = Field(description="The identifier of the evaluation group.")
    id: str = Field(description="Unique identifier for the evaluation run output item.")
    object: Literal["eval.run.output_item"] = Field(
        default="eval.run.output_item",
        description="The type of the object. Always 'eval.run.output_item'.",
    )
    results: List[Dict] = Field(
        description="A list of results from the evaluation run."
    )
    run_id: str = Field(
        description="The identifier of the evaluation run associated with this output item."
    )
    sample: EvalRunOutputItemSampleSchema = Field(
        description="A sample containing the input and output of the evaluation run."
    )
    status: str = Field(description="The status of the evaluation run.")
