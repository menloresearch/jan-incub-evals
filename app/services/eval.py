from ..schemas.eval import EvalConfig, EvalResponse


class Eval:
    """
    An Eval object with a data source config and testing criteria. An Eval represents a task to be done for your LLM integration. Like:
        - Improve the quality of my chatbot
        - See how well my chatbot handles customer support
        - Check if o4-mini is better at my usecase than gpt-4o
    """

    def __init__(self, eval_id: str, created_at: int, config: EvalConfig):
        self.id = eval_id
        self.created_at = created_at
        self.object = "eval"
        self.data_source_config = config.data_source_config
        self.metadata = config.metadata
        self.name = config.name
        self.testing_criteria = config.testing_criteria

    def to_response(self) -> EvalResponse:
        """Convert to EvalResponse for API responses"""
        return EvalResponse(
            id=self.id,
            created_at=self.created_at,
            object=self.object,
            data_source_config=self.data_source_config,
            metadata=self.metadata,
            name=self.name,
            testing_criteria=self.testing_criteria,
        )
