# Kushim: A Framework for Verifiable LLM Evaluation Datasets

Kushim is a framework for generating high-quality, verifiable Question & Answer datasets from Wikipedia articles. It provides a complete, end-to-end workflow from data generation to packaging and sharing on the Hugging Face Hub.

## Installation

This project uses `uv` for package management.

1.  **Install `uv`**:
    If you don't have `uv`, install it first:
    ```bash
    pip install uv
    ```

2.  **Create a Virtual Environment and Install Dependencies**:
    The `pyproject.toml` file lists all necessary dependencies. `uv` can install them for you.
    ```bash
    uv pip install -r requirements.txt
    ```

3.  **Set Up Your Language Model**:
    The generation pipeline uses a Large Language Model via `dspy`. You must configure your API keys in a `.env` file in the project root. For example:
    ```.env
    OPENAI_API_KEY="sk-..."
    GROQ_API_KEY="gsk_..."
    ```

## End-to-End Data Workflow

The following scripts in the `examples/` directory are designed to be run in order to take you from initial idea to a shared, high-quality dataset.

### Step 1: Generate Raw Datasets

First, generate individual Q&A datasets on various topics.

-   **Script**: `examples/wikiqa.py`
-   **What it does**: Fetches content from Wikipedia based on a search query (e.g., "Sony"), generates Q&A pairs, validates them, and saves the results as `_qa_dataset.csv` and `_sources.json` files in the `datasets/` directory.
-   **How to use**: Open the script and change the `search_query` variable to your topic of interest. Then run it:
    ```bash
    uv run examples/wikiqa.py
    ```
    Repeat this process for as many topics as you need.

### Step 2: Finalize the Master Dataset

Once you have generated several raw datasets, merge them into a single, master dataset.

-   **Script**: `examples/finalize_dataset.py`
-   **What it does**: Merges all `_qa_dataset.csv` and `_sources.json` files, randomly shuffles the combined data for unbiased distribution, and saves the result as `final_shuffled_qa_dataset.csv` and `final_sources.json`.
-   **How to use**:
    ```bash
    uv run examples/finalize_dataset.py
    ```

### Step 3: Package and Share on Hugging Face Hub

The final step is to prepare your dataset for public (or private) release. This script creates a small, unencrypted public sample and a large, securely encrypted main file.

-   **Script**: `examples/prepare_and_push_dataset.py`
-   **What it does**:
    1.  Loads your final, shuffled dataset.
    2.  Creates a small, public sample (e.g., 200 rows).
    3.  Encrypts the remaining data.
    4.  Pushes the public sample, the encrypted main file, and the sources file to the Hugging Face Hub.
-   **How to use**:
    1.  **Log in to Hugging Face**: Run the login command within the `uv` environment. It will prompt you for an access token with `write` permissions.
        ```bash
        uv run huggingface-cli login
        ```
    2.  **Configure Script**: Edit the `examples/prepare_and_push_dataset.py` script to set your `repo_id` (e.g., `"your-username/your-dataset-name"`).
    3.  **Run the Script**:
        ```bash
        uv run examples/prepare_and_push_dataset.py
        ```

## Utilities

The `kushim/utils/` directory contains the core, reusable functions that power the example scripts. This includes modules for merging, sampling, encrypting, and uploading datasets. You can import and use these in your own custom Python scripts for more advanced workflows.
