## MARL API

This directory contains a FastAPI application to handle the training and inference for the Multi-Agent Reinforcement Learning (MARL) model used in the Capture the Flag simulation.

### Setup

1.  Navigate to this directory: `cd marl-api`
2.  Create and activate a virtual environment (using uv):
    ```bash
    uv venv
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

### Running the API

Use uvicorn to run the development server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### Structure

-   `main.py`: The main FastAPI application entry point.
-   `requirements.txt`: Python dependencies.
-   *(Add other modules/directories as the project grows, e.g., for environment definition, agent models, training logic)* 