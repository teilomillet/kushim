import os
import json
import polars as pl
import logging
from typing import List, Dict, Tuple, Optional

def _merge_qa_dataframes(list_of_dfs: List[pl.DataFrame]) -> pl.DataFrame:
    """
    Concatenates a list of Polars DataFrames into a single DataFrame.
    
    This is a pure data transformation function.
    """
    if not list_of_dfs:
        return pl.DataFrame()
    return pl.concat(list_of_dfs)

def _deduplicate_and_merge_sources(list_of_source_lists: List[List[Dict]]) -> List[Dict]:
    """
    Merges multiple lists of source dictionaries, ensuring uniqueness based on title.

    This function isolates the core logic of deduplication.
    """
    all_sources = []
    seen_source_titles = set()
    for source_list in list_of_source_lists:
        for source in source_list:
            if 'title' in source and source['title'] not in seen_source_titles:
                all_sources.append(source)
                seen_source_titles.add(source['title'])
    return all_sources

def _find_dataset_files(
    input_dir: str,
    qa_filename_to_exclude: str,
    sources_filename_to_exclude: str
) -> Tuple[List[str], List[str]]:
    """Finds all CSV QA datasets and JSON source files in a directory."""
    try:
        files = os.listdir(input_dir)
        csv_files = [
            os.path.join(input_dir, f) for f in files 
            if f.endswith('_qa_dataset.csv') and f != qa_filename_to_exclude
        ]
        json_files = [
            os.path.join(input_dir, f) for f in files 
            if f.endswith('_sources.json') and f != sources_filename_to_exclude
        ]
        return csv_files, json_files
    except FileNotFoundError:
        logging.error(f"Input directory not found: {input_dir}")
        return [], []

def merge_datasets(
    input_dir: str,
    merged_qa_filename: str = "merged_qa_dataset.csv",
    merged_sources_filename: str = "merged_sources.json",
    save_merged: bool = True
) -> Tuple[Optional[pl.DataFrame], List[Dict]]:
    """
    Merges all individual QA datasets and source files from a directory
    into a single dataset. It can optionally save the merged files and
    always returns them.

    This function orchestrates the process by finding files, calling
    data transformation functions, and handling I/O and logging.

    Args:
        input_dir (str): The path to the directory containing the dataset files.
        merged_qa_filename (str): The filename for the merged CSV of Q&A pairs.
        merged_sources_filename (str): The filename for the merged JSON of sources.
        save_merged (bool): If True, saves the merged files to disk.

    Returns:
        A tuple containing:
        - A polars DataFrame with the merged Q&A pairs (or None if no data).
        - A list of dictionaries with the unique source article data.
    """
    logging.info(f"Starting dataset merge process for directory: {input_dir}")

    csv_files, json_files = _find_dataset_files(
        input_dir,
        qa_filename_to_exclude=merged_qa_filename,
        sources_filename_to_exclude=merged_sources_filename
    )

    logging.info(f"Found {len(csv_files)} CSV files to merge.")
    all_qa_dfs = []
    for filepath in csv_files:
        try:
            df = pl.read_csv(filepath)
            all_qa_dfs.append(df)
            logging.info(f"Loaded {os.path.basename(filepath)} with {len(df)} rows.")
        except Exception as e:
            logging.error(f"Could not read or process CSV file {os.path.basename(filepath)}: {e}")

    merged_qa_df = None
    if all_qa_dfs:
        merged_qa_df = _merge_qa_dataframes(all_qa_dfs)
        if save_merged:
            output_qa_path = os.path.join(input_dir, merged_qa_filename)
            try:
                merged_qa_df.write_csv(output_qa_path)
                logging.info(f"Successfully merged {len(all_qa_dfs)} QA datasets into {output_qa_path} with a total of {len(merged_qa_df)} rows.")
            except Exception as e:
                logging.error(f"Failed to write merged QA dataset to {output_qa_path}: {e}")
    else:
        logging.warning("No QA datasets found to merge.")

    logging.info(f"Found {len(json_files)} JSON source files to merge.")
    list_of_source_lists = []
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sources = json.load(f)
                list_of_source_lists.append(sources)
                logging.info(f"Loaded {os.path.basename(filepath)} containing {len(sources)} sources.")
        except Exception as e:
            logging.error(f"Could not read or process JSON file {os.path.basename(filepath)}: {e}")
    
    merged_sources = []
    if list_of_source_lists:
        merged_sources = _deduplicate_and_merge_sources(list_of_source_lists)
        if save_merged:
            output_sources_path = os.path.join(input_dir, merged_sources_filename)
            try:
                with open(output_sources_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_sources, f, ensure_ascii=False, indent=4)
                logging.info(f"Successfully merged sources into {output_sources_path}, with {len(merged_sources)} unique sources.")
            except Exception as e:
                logging.error(f"Failed to write merged sources to {output_sources_path}: {e}")
    else:
        logging.warning("No source files found to merge.")
    
    return merged_qa_df, merged_sources

def sample_and_split_dataset(
    df: pl.DataFrame,
    sample_size: int | float,
    seed: int | None = None
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splits a DataFrame into a random sample and the remaining data.

    This is useful for creating a public preview of a dataset while keeping
    the main body of data separate. It ensures no rows are duplicated
    between the two output DataFrames.

    Args:
        df (pl.DataFrame): The input DataFrame to split.
        sample_size (int | float): The size of the sample. If an integer,
                                   it's the number of rows. If a float,
                                   it's the fraction of rows.
        seed (int, optional): A random seed for reproducibility.

    Returns:
        A tuple containing two DataFrames: (sample_df, main_df).
    """
    if not isinstance(df, pl.DataFrame) or df.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    # Ensure the sample size is valid
    if isinstance(sample_size, float) and not (0.0 < sample_size < 1.0):
        raise ValueError("Sample size fraction must be between 0.0 and 1.0.")
    if isinstance(sample_size, int) and sample_size >= len(df):
        logging.warning("Sample size is >= total rows. The main dataset will be empty.")
        return df.clone(), pl.DataFrame()

    logging.info(f"Splitting dataset: creating a random sample of size {sample_size}.")

    # Create a temporary column with unique IDs to perform an anti-join
    # which is a robust way to find the rows not in the sample.
    with_id = df.with_row_count("unique_temp_id")

    sample_df_with_id = with_id.sample(
        n=sample_size if isinstance(sample_size, int) else None,
        fraction=sample_size if isinstance(sample_size, float) else None,
        shuffle=True,
        seed=seed
    )

    # The main DataFrame is everything not in the sample (anti-join)
    main_df_with_id = with_id.join(
        sample_df_with_id, on="unique_temp_id", how="anti"
    )

    # Drop the temporary ID column from both frames
    sample_df = sample_df_with_id.drop("unique_temp_id")
    main_df = main_df_with_id.drop("unique_temp_id")
    
    logging.info(f"Split complete. Sample size: {len(sample_df)} rows. Main dataset size: {len(main_df)} rows.")

    return sample_df, main_df 