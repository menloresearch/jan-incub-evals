from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from app.schemas.eval import EvalConfig, EvalResponse, EvalDeleteResponse
from app.schemas.eval_run import EvalRunConfig, EvalRunResponse

router = APIRouter()


@router.post("/evals", response_model=EvalResponse)
async def create_eval(eval_data: EvalConfig, request: Request):
    """
    Create a new evaluation
    """
    eval_obj = request.app.state.eval_service.create_eval(eval_data)
    return eval_obj.to_response()


@router.get("/evals/{eval_id}", response_model=EvalResponse)
async def get_eval(eval_id: str, request: Request):
    """
    Retrieve an evaluation by ID
    """
    eval_obj = request.app.state.eval_service.get_eval(eval_id)
    if not eval_obj:
        raise HTTPException(status_code=404, detail="Eval not found")
    return eval_obj.to_response()


@router.post("/evals/{eval_id}", response_model=EvalResponse)
async def update_eval(eval_id: str, eval_data: EvalConfig, request: Request):
    """
    Update an existing evaluation by ID (partial updates supported)
    """
    updated_eval = request.app.state.eval_service.update_eval(eval_id, eval_data)
    if not updated_eval:
        raise HTTPException(status_code=404, detail="Eval not found")
    return updated_eval.to_response()


@router.delete("/evals/{eval_id}", response_model=EvalDeleteResponse)
async def delete_eval(eval_id: str, request: Request):
    """
    Delete an evaluation by ID
    """
    success = request.app.state.eval_service.delete_eval(eval_id)
    if not success:
        raise HTTPException(status_code=404, detail="Eval not found")
    return EvalDeleteResponse(deleted=True, eval_id=eval_id)


@router.post("/evals/{eval_id}/runs", response_model=EvalRunResponse)
async def create_eval_run(
    eval_id: str,
    run_data: EvalRunConfig,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Kicks off a new run for a given evaluation, specifying the data source,
    and what model configuration to use to test. The datasource will be
    validated against the schema specified in the config of the evaluation.
    """
    eval_service = request.app.state.eval_service
    if not eval_service.eval_exists(eval_id):
        raise HTTPException(status_code=404, detail="Eval not found")

    eval_run = eval_service.create_eval_run(eval_id, run_data)

    # Start background processing
    background_tasks.add_task(eval_service.processor.process_eval_run, eval_run.id)

    return eval_run


@router.get("/evals/{eval_id}/runs/{run_id}", response_model=EvalRunResponse)
async def get_eval_run(eval_id: str, run_id: str, request: Request):
    """
    Retrieve an evaluation run by eval_id and run_id
    """
    eval_service = request.app.state.eval_service
    eval_run_config = eval_service.get_eval_run_config_by_eval_and_run_id(
        eval_id, run_id
    )

    if not eval_run_config:
        raise HTTPException(status_code=404, detail="Eval run not found")

    return eval_run_config
