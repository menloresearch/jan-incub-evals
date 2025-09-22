from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Dict, Annotated, Union, Any
from abc import ABC
from .grader import GraderConfigUnion


# Schema validation class
class ItemSchema(BaseModel):
    type: str
    properties: Dict[str, Dict[str, Any]]
    required: List[str]

    @field_validator("required")
    @classmethod
    def validate_required_subset_of_properties(cls, v, info):
        if info.data and "properties" in info.data:
            properties_keys = set(info.data["properties"].keys())
            required_set = set(v)
            if not required_set.issubset(properties_keys):
                raise ValueError("required fields must be a subset of properties keys")
        return v


# Base class
class data_source_config(BaseModel, ABC):
    type: str
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Set of 16 key-value pairs for additional information",
    )


# Specific implementations
class CustomDataSourceConfig(data_source_config):
    item_schema: ItemSchema = Field(
        description="Schema definition for the data structure - property names can be customized by user"
    )
    type: Literal["custom"] = Field(
        default="custom", description="The type of data source. Always custom."
    )
    include_sample_schema: bool = Field(
        default=False,
        description="Whether the eval should expect you to populate the sample namespace (ie, by generating responses off of your data source)",
    )


class LogsDataSourceConfig(data_source_config):
    type: Literal["database"] = "database"


# Union type with discriminator
DataSourceConfigUnion = Union[CustomDataSourceConfig, LogsDataSourceConfig]


class Eval(BaseModel):
    """
    An Eval object with a data source config and testing criteria. An Eval represents a task to be done for your LLM integration. Like:
        - Improve the quality of my chatbot
        - See how well my chatbot handles customer support
        - Check if o4-mini is better at my usecase than gpt-4o
    """

    created_at: int = Field(
        description="The Unix timestamp (in seconds) for when the eval was created."
    )
    data_source_config: DataSourceConfigUnion = Field(
        discriminator="type",
        description="Configuration of data sources used in runs of the evaluation.",
    )
    id: str = Field(description="Unique identifier for the evaluation.")
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Set of 16 key-value pairs for additional information",
    )
    name: str = Field(description="The name of the evaluation.")
    object: Literal["eval"] = Field(
        default="eval", description="Object type, always 'eval' for evaluation objects"
    )
    testing_criteria: List[
        Annotated[GraderConfigUnion, Field(discriminator="type")]
    ] = Field(description="A list of testing criteria.")
