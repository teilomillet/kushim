import logging
from huggingface_hub import HfFolder
from datasets import DatasetDict
from typing import Optional

def push_dataset_dict_to_hub(
    repo_id: str,
    dataset_dict: DatasetDict,
    token: Optional[str] = None,
    private: bool = False,
):
    """
    Pushes a datasets.DatasetDict object to the Hugging Face Hub.

    This is the standard and recommended way to share a dataset, as it handles
    data formatting (Parquet), versioning, and allows for splits (e.g.,
    'train', 'test', 'sample').

    Args:
        repo_id (str): The ID of the repository (e.g., "username/dataset-name").
        dataset_dict (DatasetDict): The dataset object to upload.
        token (str, optional): Hugging Face API token.
        private (bool): Whether to make the repository private.
    """
    if token is None:
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("Hugging Face token not found. Please log in with `huggingface-cli login`.")

    logging.info(f"Uploading DatasetDict to repository: '{repo_id}'")
    try:
        # The push_to_hub method on a DatasetDict handles all the complexity
        # of creating the repo, converting to Parquet, and uploading.
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token
        )
        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        logging.info(f"Successfully pushed dataset to the Hub. View it at: {repo_url}")
        return repo_url
    except Exception as e:
        logging.error(f"Failed to push DatasetDict to the Hub: {e}")
        # Re-raising the exception allows the calling script to handle it.
        raise 