from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Base model that forbids extra fields for strict validation"""

    model_config = ConfigDict(extra="forbid")
