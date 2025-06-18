"""
This script demonstrates how to use the push_dataset_to_hub utility to
upload a merged dataset to the Hugging Face Hub.

Prerequisites:
1. You must have a Hugging Face account.
2. You must be logged in via the command line: `huggingface-cli login`
   You will be prompted to enter a token with write permissions.
"""
import os
import logging
from kushim.utils import push_dataset_to_hub

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the dataset push process.
    """
    # Configuration
    # IMPORTANT: Replace with your Hugging Face username and a desired dataset name.
    # For example: "my-username/kushim-generated-qa"
    repo_id = "YOUR_USERNAME/YOUR_DATASET_NAME"

    # The directory where your merged dataset files are located.
    datasets_dir = "datasets"
    
    # The names of the files to be uploaded.
    csv_to_upload = os.path.join(datasets_dir, "merged_qa_dataset.csv")
    sources_to_upload = os.path.join(datasets_dir, "merged_sources.json")

    # Pre-flight Check
    if "YOUR_USERNAME" in repo_id:
        logging.error("Please update the 'repo_id' in this script with your Hugging Face username and a dataset name.")
        return

    if not os.path.exists(csv_to_upload):
        logging.error(f"The dataset file was not found at {csv_to_upload}.")
        logging.error("Please run the 'examples/merge_datasets.py' script first to generate it.")
        return

    # Push to Hub
    logging.info(f"Attempting to push dataset to Hugging Face Hub repository: {repo_id}")
    
    push_dataset_to_hub(
        repo_id=repo_id,
        csv_file_path=csv_to_upload,
        sources_file_path=sources_to_upload,
        private=False  # Set to True if you want the dataset to be private
    )

if __name__ == "__main__":
    main() 