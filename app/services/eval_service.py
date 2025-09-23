from typing import Dict, Optional
from datetime import datetime
import uuid
from app.schemas.eval import EvalConfig
from app.schemas.eval_run import EvalRunConfig, EvalRunResponse, EvalRunStatus
from .eval import Eval
from .eval_run import EvalRun
from .eval_processor import EvalProcessor


class EvalService:
    def __init__(self, llm_base_url: str = None, llm_api_key: str = None):
        self._evals_storage: Dict[str, Eval] = {}
        self._runs_storage: Dict[str, EvalRun] = {}  # Store plain EvalRun objects
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.processor = EvalProcessor(self)

    def create_eval(self, eval_config: EvalConfig) -> Eval:
        eval_id = f"eval_{uuid.uuid4().hex}"
        created_at = int(datetime.now().timestamp())

        eval_obj = Eval(eval_id, created_at, eval_config)
        self._evals_storage[eval_id] = eval_obj
        return eval_obj

    def get_eval(self, eval_id: str) -> Optional[Eval]:
        return self._evals_storage.get(eval_id)

    def update_eval(self, eval_id: str, eval_config: EvalConfig) -> Optional[Eval]:
        if eval_id not in self._evals_storage:
            return None

        existing_eval = self._evals_storage[eval_id]

        # Create updated eval with same id and created_at, but new config
        updated_eval = Eval(existing_eval.id, existing_eval.created_at, eval_config)
        self._evals_storage[eval_id] = updated_eval
        return updated_eval

    def delete_eval(self, eval_id: str) -> bool:
        if eval_id not in self._evals_storage:
            return False

        del self._evals_storage[eval_id]
        return True

    def eval_exists(self, eval_id: str) -> bool:
        return eval_id in self._evals_storage

    def create_eval_run(self, eval_id: str, run_data: EvalRunConfig) -> EvalRunResponse:
        run_id = f"evalrun_{uuid.uuid4().hex}"
        created_at = int(datetime.now().timestamp())
        report_url = f"https://server.jan.ai/evaluations/{eval_id}&run_id={run_id}"

        # Create plain EvalRun for runtime processing
        eval_run = EvalRun(run_id, eval_id, created_at, report_url, run_data)
        self._runs_storage[run_id] = eval_run

        return eval_run.to_response()

    def get_eval_run(self, run_id: str) -> Optional[EvalRun]:
        """Get the runtime EvalRun object for processing."""
        return self._runs_storage.get(run_id)

    def get_eval_run_config(self, run_id: str) -> Optional[EvalRunResponse]:
        """Get the EvalRunResponse for API responses."""
        eval_run = self._runs_storage.get(run_id)
        return eval_run.to_response() if eval_run else None

    def get_eval_run_by_eval_and_run_id(
        self, eval_id: str, run_id: str
    ) -> Optional[EvalRun]:
        """
        Retrieve an eval run by both eval_id and run_id, ensuring the run belongs to the eval.
        Returns runtime EvalRun object.
        """
        # Check if eval exists
        if not self.eval_exists(eval_id):
            return None

        # Get the run
        eval_run = self.get_eval_run(run_id)
        if not eval_run:
            return None

        # Verify the run belongs to the specified eval
        if eval_run.eval_id != eval_id:
            return None

        return eval_run

    def get_eval_run_config_by_eval_and_run_id(
        self, eval_id: str, run_id: str
    ) -> Optional[EvalRunResponse]:
        """
        Retrieve an eval run config by both eval_id and run_id for API responses.
        """
        eval_run = self.get_eval_run_by_eval_and_run_id(eval_id, run_id)
        return eval_run.to_response() if eval_run else None

    def update_run_status(
        self, run_id: str, status: EvalRunStatus
    ) -> Optional[EvalRun]:
        if run_id not in self._runs_storage:
            return None

        eval_run = self._runs_storage[run_id]
        eval_run.update_status(status)
        return eval_run


# eval_service will be initialized in main.py with app state values
