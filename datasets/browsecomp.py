import asyncio
import os
import polars as pl
from kushim.utils import decrypt_file

async def main():
    """
    Main function to orchestrate the download and decryption of the
    BrowseComp dataset using an adaptive processing utility.
    """
    # URL for the dataset
    url = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    
    print("Starting dataset processing...")
    
    # The `decrypt_file` utility now returns the output path and can
    # display a sample directly.
    output_csv = await decrypt_file(
        input_path=url,
        password_column="canary",
        columns_to_decrypt=["problem", "answer"],
        show_sample=3,
    )
    
    print(f"\nProcessing complete. Decrypted data is in: {output_csv}")
    
    # Optionally, you can still perform further analysis on the output file
    if os.path.exists(output_csv):
        # Example: loading the data for other uses
        df = pl.read_csv(output_csv)
        print(f"\nSuccessfully loaded {len(df)} rows for further use.")

if __name__ == "__main__":
    # In Python, top-level await is not always supported, so we run
    # the async main function using asyncio.run().
    asyncio.run(main()) 