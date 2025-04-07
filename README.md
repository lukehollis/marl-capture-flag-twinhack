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

First, choose what version / model you want to develop with and the locations you have them on your device. Because this was just a hackathon project, I didn't do anything fancier with handling the models, so configure these variables: 

```bash
# --- Configuration ---
MODEL_EPISODE = 200 # Choose which saved models to load
VERSION = "v7"
MODEL_DIR = os.path.expanduser(f"~/torch_results/ctf_iac_{VERSION}")
```

Then use uvicorn to run the development server:

```bash
uvicorn main:app --reload
```

or also 

```bash
python main.py
```


The API will be available at `http://127.0.0.1:8000`.

### Structure

-   `main.py`: The main FastAPI application entry point.
-   `requirements.txt`: Python dependencies.
-   *(Add other modules/directories as the project grows, e.g., for environment definition, agent models, training logic)* 