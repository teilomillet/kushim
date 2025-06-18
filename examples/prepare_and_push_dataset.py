"""
This script provides a complete, end-to-end workflow for preparing and
sharing a high-quality dataset on the Hugging Face Hub in the standard
'datasets' format.

The process includes:
1. Loading the final, shuffled QA dataset and its sources.
2. Splitting the data into a public 'sample' split and a main 'eval' split.
3. Encrypting the 'eval' split in-memory.
4. Packaging 'sample', 'eval', and 'sources' into a single DatasetDict object.
5. Pushing the complete DatasetDict to the Hub.
"""
import os
import logging
import polars as pl
import json
from kushim.utils import (
    sample_and_split_dataset,
    encrypt_dataframe,
    push_dataset_dict_to_hub,
)
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to orchestrate the dataset preparation and upload workflow."""
    
    # Configuration
    repo_id = "teilomillet/wikipeqa"
    datasets_dir = "datasets"
    
    # Input files created by `finalize_dataset.py`
    final_qa_path = os.path.join(datasets_dir, "final_shuffled_qa_dataset.csv")
    final_sources_path = os.path.join(datasets_dir, "final_sources.json")

    # The size of the public, unencrypted sample.
    sample_size = 200
    
    # Pre-flight Check
    if "YOUR_USERNAME" in repo_id:
        logging.error("Please update the 'repo_id' in this script with your Hugging Face username and a dataset name.")
        return
    if not os.path.exists(final_qa_path) or not os.path.exists(final_sources_path):
        logging.error(f"Final dataset files not found. Please run 'finalize_dataset.py' first.")
        return

    # Step 1: Load Final Datasets
    logging.info(f"--- Step 1: Loading final dataset from '{final_qa_path}' ---")
    full_df = pl.read_csv(final_qa_path)
    with open(final_sources_path, 'r', encoding='utf-8') as f:
        sources_data = json.load(f)
    logging.info(f"Loaded {len(full_df)} Q&A pairs and {len(sources_data)} source articles.")

    # Step 2: Create Sample and Eval Splits
    logging.info(f"--- Step 2: Splitting data into sample ({sample_size} rows) and eval splits ---")
    sample_df, eval_df = sample_and_split_dataset(full_df, sample_size=sample_size, seed=42)

    # Step 3: Encrypt the Eval Split
    logging.info(f"--- Step 3: Encrypting the 'eval' split in-memory ---")
    # Derive a meaningful canary name from the repository ID.
    canary_name = repo_id.split('/')[-1]
    encrypted_eval_df = encrypt_dataframe(
        df=eval_df,
        columns_to_encrypt=['question', 'answer', 'source'],
        canary_name=canary_name
    )
    
    # To ensure schema consistency for the Hugging Face upload, we add a placeholder
    # 'canary' column to the public 'sample' split. This column is present in the
    # encrypted 'eval' split but is not needed for the sample, so we fill it with
    # a benign null value (an empty string).
    sample_df = sample_df.with_columns(
        pl.lit("").alias("canary")
    )

    # Step 4: Package into a DatasetDict
    logging.info("--- Step 4: Creating Hugging Face DatasetDict ---")
    
    # The main dataset contains the Q&A pairs, split into a public sample
    # and an encrypted evaluation set. The 'sources' data will be uploaded
    # separately to the same repository, as it has a different schema.
    dataset_dict = DatasetDict({
        "sample": Dataset.from_polars(sample_df),
        "eval": Dataset.from_polars(encrypted_eval_df),
    })
    
    logging.info("DatasetDict created with two splits: 'sample' and 'eval'.")
    print(dataset_dict)

    # Step 5: Push to Hugging Face Hub
    logging.info("--- Step 5: Pushing DatasetDict and sources to the Hub ---")
    
    # This step requires you to be logged in via `huggingface-cli login`
    try:
        # First, push the main DatasetDict containing the 'sample' and 'eval' splits.
        push_dataset_dict_to_hub(
            repo_id=repo_id,
            dataset_dict=dataset_dict,
            private=False
        )

        # Next, upload the `final_sources.json` file directly to the root of the
        # same repository. This keeps the source data alongside the Q&A data
        # without causing schema conflicts.
        logging.info(f"Uploading '{os.path.basename(final_sources_path)}' to '{repo_id}'...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=final_sources_path,
            path_in_repo="sources.json",  # Upload as 'sources.json'
            repo_id=repo_id,
            repo_type="dataset",
            token=HfFolder.get_token(),
        )
        logging.info(f"Successfully uploaded '{os.path.basename(final_sources_path)}'.")

    except Exception as e:
        logging.error(f"An error occurred while pushing to the Hub: {e}")

if __name__ == "__main__":
    main() 