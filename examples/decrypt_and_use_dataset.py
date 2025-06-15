"""
This script demonstrates how to download a dataset from the Hugging Face Hub,
decrypt the 'eval' split, and use the data.

It performs the following steps:
1.  Specifies the repository ID of the dataset on the Hub.
2.  Loads the DatasetDict from the Hub.
3.  Extracts the 'eval' split, which contains encrypted data.
4.  Converts the 'eval' split from a Hugging Face Dataset to a Polars DataFrame.
5.  Calls the `decrypt_dataframe` utility to decrypt the specified columns.
6.  Displays the head of the decrypted DataFrame to verify the process.
"""
import logging
import polars as pl
from datasets import load_dataset
from kushim.utils import decrypt_dataframe

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to orchestrate the dataset download and decryption."""
    
    repo_id = "teilomillet/wikiqa"
    
    logging.info(f"--- Step 1: Loading dataset '{repo_id}' from the Hugging Face Hub ---")
    try:
        dataset_dict = load_dataset(repo_id)
        logging.info("Dataset loaded successfully.")
        print(dataset_dict)
    except Exception as e:
        logging.error(f"Failed to load dataset. Ensure you are logged in (`huggingface-cli login`) and the repository exists. Error: {e}")
        return

    # === Step 2: Extract and Decrypt the 'eval' Split ===
    if 'eval' not in dataset_dict:
        logging.error("The 'eval' split was not found in the loaded dataset.")
        return
        
    logging.info("--- Step 2: Decrypting the 'eval' split ---")
    
    # Convert the Hugging Face Dataset to a Polars DataFrame for processing.
    eval_dataset = dataset_dict['eval']
    encrypted_df = eval_dataset.to_polars()
    
    # Decrypt the relevant columns. The 'canary' column is used for decryption
    # and is automatically dropped by the function.
    decrypted_df = decrypt_dataframe(
        df=encrypted_df,
        columns_to_decrypt=['question', 'answer', 'source']
    )

    logging.info("--- Step 3: Verification ---")
    logging.info("Decryption complete. Showing the first 5 rows of the decrypted 'eval' data:")
    print(decrypted_df.head())

if __name__ == "__main__":
    main() 