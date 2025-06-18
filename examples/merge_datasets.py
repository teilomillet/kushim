"""
This script demonstrates how to use the merge_datasets utility to combine
multiple Q&A datasets into a single, unified dataset.
"""
import os
import logging
from kushim.utils import merge_datasets

# Configure basic logging to see the output from the merge utility.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the dataset merging process.
    """
    # The directory where your individual datasets are stored.
    # This is the same directory where the merged files will be saved.
    datasets_dir = "datasets"

    logging.info(f"Attempting to merge datasets in '{datasets_dir}'...")
    
    # Call the utility function to perform the merge.
    # The function will automatically find all `*_qa_dataset.csv` and
    # `*_sources.json` files in the specified directory, save the
    # merged results, and return the merged data.
    merged_df, merged_sources = merge_datasets(input_dir=datasets_dir)
    
    logging.info("Merge process completed.")

    # Display Dataset Summary
    if merged_df is not None and not merged_df.is_empty():
        print("\\n--- Merged Dataset Summary ---")
        rows, cols = merged_df.shape
        print(f"Dimensions: {rows} rows, {cols} columns")
        print(f"Column Names: {merged_df.columns}")
        print("\\nFirst 5 rows of data:")
        print(merged_df.head())
    else:
        print("\\nNo data was merged, so no summary can be displayed.")
    
    if merged_sources:
        print(f"\\nTotal unique source documents merged: {len(merged_sources)}")

    print(f"\\nFull merged files are saved in the '{datasets_dir}' directory.")

if __name__ == "__main__":
    main() 