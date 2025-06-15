"""
This script provides a streamlined workflow for finalizing a dataset.

It performs the following steps:
1. Merges all individual `_qa_dataset.csv` files from the 'datasets' directory.
2. Randomly shuffles the entire merged dataset to ensure unbiased data distribution.
3. Saves the result as a single, final CSV file.
4. Prints a summary of the final product.
"""
import os
import logging
import polars as pl
from kushim.utils import merge_datasets
import json

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the dataset finalization workflow.
    """
    # --- Configuration ---
    datasets_dir = "datasets"
    final_qa_filename = "final_shuffled_qa_dataset.csv"
    final_sources_filename = "final_sources.json"
    final_qa_path = os.path.join(datasets_dir, final_qa_filename)
    final_sources_path = os.path.join(datasets_dir, final_sources_filename)
    
    # A seed for the random shuffle to ensure reproducibility
    random_seed = 42

    # === Step 1: Merge All Datasets ===
    logging.info(f"--- Step 1: Merging all datasets from '{datasets_dir}' ---")
    
    # We call merge_datasets with `save_merged=False` to prevent it from
    # creating intermediate files. We only want the final, shuffled output.
    merged_df, merged_sources = merge_datasets(input_dir=datasets_dir, save_merged=False)
    
    if merged_df is None or merged_df.is_empty():
        logging.error("Merging failed or produced no data. Cannot create a final dataset.")
        return
        
    logging.info(f"Merging complete. The full dataset has {len(merged_df)} rows and {len(merged_sources)} unique source articles.")

    # === Step 2: Shuffle the Merged Dataset ===
    logging.info(f"--- Step 2: Shuffling the dataset with random seed {random_seed} ---")
    
    # Use the sample() method with a fraction of 1.0 to perform a full,
    # efficient, in-memory shuffle of the DataFrame.
    shuffled_df = merged_df.sample(fraction=1.0, shuffle=True, seed=random_seed)
    
    logging.info("Dataset has been successfully shuffled.")

    # === Step 3: Save the Final Dataset and Sources ===
    logging.info(f"--- Step 3: Saving final files ---")
    
    try:
        # Save the shuffled Q&A data
        shuffled_df.write_csv(final_qa_path)
        logging.info(f"Final shuffled Q&A dataset saved to '{final_qa_path}'")
        
        # Save the consolidated source articles
        if merged_sources:
            with open(final_sources_path, 'w', encoding='utf-8') as f:
                json.dump(merged_sources, f, ensure_ascii=False, indent=4)
            logging.info(f"Final sources file saved to '{final_sources_path}'")

    except Exception as e:
        logging.error(f"Failed to save the final dataset files: {e}")
        return

    # === Step 4: Display Final Dataset Summary ===
    print("\\n--- Final Shuffled Dataset Summary ---")
    rows, cols = shuffled_df.shape
    print(f"Dimensions: {rows} rows, {cols} columns")
    print(f"Column Names: {shuffled_df.columns}")
    print("\\nFirst 5 rows of the final, shuffled data:")
    print(shuffled_df.head())
    print(f"\\nFull shuffled Q&A dataset saved to: {final_qa_path}")
    print(f"Full sources file saved to: {final_sources_path}")

if __name__ == "__main__":
    main() 