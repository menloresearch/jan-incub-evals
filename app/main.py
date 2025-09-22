import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from app.api.v1.endpoints import router as v1_router
from app.services.eval_service import EvalService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI(
    title="Eval Microservice",
    description="API for creating and managing evaluations",
    version="1.0.0",
)

# Initialize configuration from environment
llm_base_url = os.getenv("LLM_BASE_URL")
llm_api_key = os.getenv("LLM_API_KEY")
print(llm_base_url)
print(llm_api_key)

# Initialize eval service with configuration
app.state.eval_service = EvalService(llm_base_url=llm_base_url, llm_api_key=llm_api_key)

app.include_router(v1_router, prefix="/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
