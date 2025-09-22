# Eval Microservice

A FastAPI-based microservice for creating and managing model evaluations. This service provides APIs to create evaluations, configure evaluation runs, and process them using various grading mechanisms.

## Features

- **Evaluation Management**: Create, read, update, and delete evaluations
- **Evaluation Runs**: Execute evaluations with configurable data sources and models
- **Background Processing**: Asynchronous evaluation run processing
- **Multiple Graders**: Support for different grading mechanisms
- **MCP Integration**: Model Context Protocol support for agent-based evaluations
- **OpenAI Integration**: Support for OpenAI-compatible LLM endpoints

## Project Structure

```
app/
├── api/v1/           # API endpoints
├── schemas/          # Pydantic models and schemas
├── services/         # Business logic and services
├── utils.py          # Utility functions
└── main.py          # FastAPI application entry point
```

## Setup

### Prerequisites

- Python 3.12+
- Virtual environment (recommended)

### Installation

1. Clone the repository and navigate to the project directory:
```bash
cd eval_microservice
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the `app/` directory with the following variables:

```env
LLM_BASE_URL=your_llm_endpoint_url
LLM_API_KEY=your_llm_api_key
```

## Running the Service

### Development Mode
```bash
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Production Mode
```bash
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The service will be available at `http://localhost:8001`

## API Documentation

Once the service is running, you can access:
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

### Main Endpoints

- `POST /v1/evals` - Create a new evaluation
- `GET /v1/evals/{eval_id}` - Retrieve an evaluation
- `POST /v1/evals/{eval_id}` - Update an evaluation
- `DELETE /v1/evals/{eval_id}` - Delete an evaluation
- `POST /v1/evals/{eval_id}/runs` - Create and start an evaluation run
- `GET /v1/evals/{eval_id}/runs/{run_id}` - Get evaluation run status

## Testing

Run tests using pytest:
```bash
pytest
```

## Architecture

### Core Components

- **EvalService**: Main service for managing evaluations and runs
- **EvalProcessor**: Handles background processing of evaluation runs
- **Graders**: Various grading mechanisms for evaluation results
- **Agent**: MCP-based agent integration for complex evaluations
- **Schemas**: Pydantic models for data validation and serialization