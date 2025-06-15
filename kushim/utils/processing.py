"""
This module provides robust, stateful, and memory-efficient utilities for
processing large datasets. It uses an adaptive strategy, choosing between
an in-memory approach for small files and a chunked, resumable approach
for large files. It uses Polars for high-performance, parallel data
manipulation.
"""
import asyncio
import json
import logging
import os
import shutil
import urllib.parse
import urllib.request
import uuid
from typing import Callable, List

import polars as pl
from .encryption import decrypt, encrypt

log = logging.getLogger(__name__)
DEFAULT_SIZE_THRESHOLD_GB = 5.0
CANARY_PREFIX = "BENCHMARK DATA SHOULD NEVER APPEAR AS PLAIN TEXT ONLINE. canary GUID"


async def decrypt_file(
    input_path: str,
    password_column: str,
    output_path: str | None = None,
    columns_to_decrypt: List[str] | None = None,
    state_file: str | None = None,
    chunksize: int = 10000,
    size_threshold_gb: float = DEFAULT_SIZE_THRESHOLD_GB,
    drop_password_column: bool = False,
    show_sample: int | None = None,
) -> str:
    """
    Asynchronously DECRYPTS specified columns in a large CSV file.
    
    By default, it keeps the password column in the output. Set
    `drop_password_column=True` to remove it.

    If `show_sample` is set to an integer, it will display that many
    rows from the decrypted file upon completion.

    Returns the path to the decrypted output file.
    """
    return await _process_file(
        input_path=input_path,
        password_column=password_column,
        output_path=output_path,
        columns_to_process=columns_to_decrypt,
        state_file=state_file,
        chunksize=chunksize,
        func_to_apply=decrypt,
        size_threshold_gb=size_threshold_gb,
        drop_password_column=drop_password_column,
        show_sample=show_sample,
    )


async def encrypt_file(
    input_path: str,
    output_path: str | None = None,
    canary_name: str | None = None,
    columns_to_encrypt: List[str] | None = None,
    state_file: str | None = None,
    chunksize: int = 10000,
    size_threshold_gb: float = DEFAULT_SIZE_THRESHOLD_GB,
    include_canary_prefix: bool = True,
) -> str:
    """
    Asynchronously ENCRYPTS specified columns in a large CSV file.
    
    If `canary_name` is not specified, it will be derived from the input filename.
    The canary prefix can be optionally excluded.

    Returns the path to the encrypted output file.
    """
    # For encryption, the password column is always the 'canary' column we create.
    password_column = "canary"

    # If no canary name is provided, derive it from the input filename.
    if canary_name is None:
        canary_name = os.path.splitext(os.path.basename(input_path))[0]
        log.info(f"No canary name provided. Defaulting to '{canary_name}' based on the input filename.")
    
    # Generate a single GUID for the entire run.
    run_guid = uuid.uuid4()
    
    canary_base = f"{canary_name}:{run_guid}"
    canary_string = f"{CANARY_PREFIX} {canary_base}" if include_canary_prefix else canary_base

    return await _process_file(
        input_path=input_path,
        password_column=password_column,
        output_path=output_path,
        columns_to_process=columns_to_encrypt,
        state_file=state_file,
        chunksize=chunksize,
        func_to_apply=encrypt,
        size_threshold_gb=size_threshold_gb,
        canary_string_to_add=canary_string,
    )


async def _process_file(
    input_path: str,
    password_column: str,
    output_path: str | None,
    columns_to_process: List[str] | None,
    state_file: str | None,
    chunksize: int,
    func_to_apply: Callable[[str, str], str],
    size_threshold_gb: float,
    canary_string_to_add: str | None = None,
    drop_password_column: bool = False,
    show_sample: int | None = None,
) -> str:
    """Generic async file processor that calls the synchronous dispatcher."""
    return await asyncio.to_thread(
        _process_file_sync,
        input_path,
        password_column,
        output_path,
        columns_to_process,
        state_file,
        chunksize,
        func_to_apply,
        size_threshold_gb,
        canary_string_to_add,
        drop_password_column,
        show_sample,
    )


def _get_input_size(input_path: str) -> int | None:
    """Gets the size of the input, whether a local file or a remote URL."""
    # This function now expects a string. The calling function `_process_file_sync`
    # is responsible for ensuring this.
    if str(input_path).startswith(('http://', 'https://')):
        try:
            req = urllib.request.Request(input_path, method='HEAD')
            with urllib.request.urlopen(req) as response:
                return int(response.headers['Content-Length'])
        except Exception as e:
            log.warning(f"Could not determine size of remote file: {e}. Defaulting to chunked processing.")
            return None
    else:
        try:
            return os.path.getsize(input_path)
        except OSError as e:
            log.warning(f"Could not determine size of local file: {e}. Defaulting to chunked processing.")
            return None


def _process_file_sync(
    input_path: str,
    password_column: str,
    output_path: str | None,
    columns_to_process: List[str] | None,
    state_file: str | None,
    chunksize: int,
    func_to_apply: Callable[[str, str], str],
    size_threshold_gb: float,
    canary_string_to_add: str | None = None,
    drop_password_column: bool = False,
    show_sample: int | None = None,
) -> str:
    """
    Synchronous dispatcher that chooses the processing strategy
    based on file size.
    """
    # --- Path Management ---
    # Logic for determining output paths is now consolidated here to ensure
    # all paths are strings and handled consistently.
    str_input_path = str(input_path)

    if output_path:
        effective_output_path = str(output_path)
    else:
        suffix = ".decrypted.csv" if func_to_apply == decrypt else ".encrypted.csv"
        if str_input_path.startswith(('http://', 'https://')):
            filename = os.path.basename(urllib.parse.urlparse(str_input_path).path)
            base, _ = os.path.splitext(filename)
            effective_output_path = f"{base}{suffix}"
        else:
            base, _ = os.path.splitext(str_input_path)
            effective_output_path = f"{base}{suffix}"
        log.info(f"No output path provided. Defaulting to '{effective_output_path}'.")

    effective_state_file = str(state_file) if state_file else f"{effective_output_path}.state.json"
    
    # --- Column Management ---
    if columns_to_process is None:
        log.info("No columns specified; processing all columns except the password column.")
        try:
            all_columns = pl.scan_csv(str_input_path).collect_schema().names()
            if func_to_apply == decrypt:
                columns_to_process = [col for col in all_columns if col != password_column]
            else: # encrypt
                columns_to_process = all_columns
            log.info(f"Auto-detected columns to process: {columns_to_process}")
        except Exception as e:
            log.error(f"Could not auto-detect columns from {str_input_path}. Please specify them manually. Error: {e}")
            raise

    # --- Strategy Selection ---
    input_size = _get_input_size(str_input_path)
    size_threshold_bytes = size_threshold_gb * (1024**3)

    if input_size is not None and input_size < size_threshold_bytes:
        log.info(f"File size ({input_size / (1024**2):.2f} MB) is below threshold. Using fast in-memory processing.")
        _process_in_memory(str_input_path, effective_output_path, password_column, func_to_apply, columns_to_process, canary_string_to_add, drop_password_column)
    else:
        log.info("File size is large or could not be determined. Using robust chunked processing.")
        _process_chunked(str_input_path, effective_output_path, password_column, effective_state_file, chunksize, func_to_apply, columns_to_process, canary_string_to_add, drop_password_column)
    
    # --- Final Actions ---
    if show_sample and show_sample > 0:
        _show_sample_data(effective_output_path, show_sample)

    return effective_output_path


def _process_in_memory(
    input_path: str,
    output_path: str,
    password_column: str,
    func_to_apply: Callable[[str, str], str],
    columns_to_process: List[str],
    canary_string_to_add: str | None = None,
    drop_password_column: bool = False,
):
    """Processes the entire file in memory. Fast but requires more RAM."""
    df = pl.read_csv(input_path)
    
    if canary_string_to_add:
        df = df.with_columns(pl.lit(canary_string_to_add).alias(password_column))

    process_expressions = [
        pl.struct([col, password_column]).map_elements(
            lambda s, c=col: func_to_apply(s[c], s[password_column]),
            return_dtype=pl.String
        ).alias(col) for col in columns_to_process
    ]
    
    df = df.with_columns(process_expressions)

    if drop_password_column and password_column in df.columns:
        df = df.drop(password_column)

    df.write_csv(output_path)
    log.info(f"In-memory processing complete. Output saved to '{output_path}'.")


def _load_state(state_file: str) -> int:
    """Loads the processing state (the last successfully processed line)."""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            return state.get('last_processed_offset', 0)
    return 0


def _save_state(state_file: str, offset: int):
    """Save the last processed offset to the state file."""
    with open(state_file, 'w') as f:
        json.dump({'last_processed_offset': offset}, f)


def _show_sample_data(output_path: str, num_samples: int):
    """
    Reads the specified output file and displays a sample of the data.
    This is a user-facing utility function to provide immediate feedback.
    """
    if not os.path.exists(output_path):
        log.warning(f"Cannot show sample. Output file not found at '{output_path}'.")
        return
        
    try:
        df = pl.read_csv(output_path)
        
        print(f"\n=== Sample of Processed Data (from {output_path}) ===")
        num_samples = min(num_samples, len(df))
        if num_samples == 0:
            print("No data to display.")
            return

        problem_col = next((c for c in df.columns if "problem" in c.lower()), df.columns[0] if df.columns else None)
        answer_col = next((c for c in df.columns if "answer" in c.lower()), df.columns[1] if len(df.columns) > 1 else None)

        for i, row in enumerate(df.head(num_samples).iter_rows(named=True), start=1):
            print(f"\n--- Sample Row {i} ---")
            if problem_col:
                problem_text = str(row.get(problem_col, ''))
                print(f"{problem_col}: {problem_text[:200]}{'...' if len(problem_text) > 200 else ''}")
            if answer_col:
                answer_text = str(row.get(answer_col, ''))
                print(f"{answer_col}: {answer_text}")
        print("-" * 50)

    except Exception as e:
        log.error(f"Could not read or display sample from '{output_path}'. Error: {e}")


def _process_chunked(
    input_path: str,
    output_path: str,
    password_column: str,
    state_file: str,
    chunksize: int,
    func_to_apply: Callable[[str, str], str],
    columns_to_process: List[str],
    canary_string_to_add: str | None = None,
    drop_password_column: bool = False,
):
    """Processes a large file in chunks to conserve memory and allow resuming."""
    temp_dir = f"{output_path}.tmp"
    os.makedirs(temp_dir, exist_ok=True)
    
    start_offset = _load_state(state_file)
    if start_offset > 0:
        log.info(f"Resuming from last saved progress at offset {start_offset}.")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            if start_offset > 0:
                f.seek(start_offset)

            while True:
                current_offset = f.tell()
                lines = [header] + f.readlines(chunksize)
                if len(lines) <= 1:
                    break

                chunk_df = pl.read_csv(bytes("".join(lines), 'utf-8'))
                
                if canary_string_to_add:
                    chunk_df = chunk_df.with_columns(pl.lit(canary_string_to_add).alias(password_column))

                process_expressions = [
                    pl.struct([col, password_column]).map_elements(
                        lambda s, c=col: func_to_apply(s[c], s[password_column]),
                        return_dtype=pl.String
                    ).alias(col) for col in columns_to_process
                ]
                
                processed_chunk = chunk_df.with_columns(process_expressions)

                chunk_output_path = os.path.join(temp_dir, f"chunk_{current_offset}.parquet")
                processed_chunk.write_parquet(chunk_output_path)
                
                _save_state(state_file, current_offset)
        
        # --- Consolidation Phase ---
        log.info("All chunks processed. Consolidating into final output file...")
        processed_files = sorted(
            [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)],
            key=lambda f: int(f.split('_')[-1].split('.')[0])
        )

        # Create a lazy scan of all Parquet files
        lazy_frames = [pl.scan_parquet(f) for f in processed_files]
        if lazy_frames:
            final_df_lazy = pl.concat(lazy_frames)
            
            # Check for column existence on the schema to avoid a PerformanceWarning
            if drop_password_column and password_column in final_df_lazy.collect_schema().names():
                final_df_lazy = final_df_lazy.drop(password_column)

            final_df_lazy.sink_csv(output_path)

        # --- Cleanup ---
        shutil.rmtree(temp_dir)
        if os.path.exists(state_file):
            os.remove(state_file)
        log.info(f"Chunked processing complete. Output is at '{output_path}'.")

    except (Exception, KeyboardInterrupt) as e:
        log.error(f"An error occurred during chunked processing: {e}")
        if isinstance(e, KeyboardInterrupt):
            log.error("The process was interrupted. You can run the script again to resume.")
        raise 


def encrypt_dataframe(
    df: pl.DataFrame,
    columns_to_encrypt: List[str],
    canary_name: str,
    include_canary_prefix: bool = True,
) -> pl.DataFrame:
    """
    Encrypts specified columns of a Polars DataFrame in memory.

    This function adds a 'canary' column which is used as the password
    for encryption, ensuring each row can be independently decrypted without
    needing to store a separate password file.

    Args:
        df (pl.DataFrame): The DataFrame to process.
        columns_to_encrypt (List[str]): A list of column names to be encrypted.
        canary_name (str): A unique name for this dataset release, used to
                           generate the encryption password.
        include_canary_prefix (bool): Whether to include the standard canary prefix.

    Returns:
        A new DataFrame with the specified columns encrypted.
    """
    log.info(f"Encrypting DataFrame in-memory with canary name: '{canary_name}'")
    password_column = "canary"
    run_guid = uuid.uuid4()
    
    canary_base = f"{canary_name}:{run_guid}"
    canary_string = f"{CANARY_PREFIX} {canary_base}" if include_canary_prefix else canary_base

    # Add the canary column to the DataFrame
    encrypted_df = df.with_columns(pl.lit(canary_string).alias(password_column))

    # Define the encryption expressions for the specified columns
    process_expressions = [
        pl.struct([col, password_column]).map_elements(
            # This lambda function applies the `encrypt` utility to each row
            lambda s, c=col: encrypt(s[c], s[password_column]),
            return_dtype=pl.String
        ).alias(col) for col in columns_to_encrypt
    ]
    
    # Apply the encryption
    encrypted_df = encrypted_df.with_columns(process_expressions)

    log.info("In-memory encryption complete.")
    return encrypted_df 


def decrypt_dataframe(
    df: pl.DataFrame,
    columns_to_decrypt: List[str],
) -> pl.DataFrame:
    """
    Decrypts specified columns of a Polars DataFrame in-memory.

    This function is the counterpart to `encrypt_dataframe`. It uses the 'canary'
    column, which contains the necessary identifier, as the password for decryption.
    After decryption, the 'canary' column is dropped.

    Args:
        df (pl.DataFrame): The DataFrame to decrypt. Must contain a 'canary' column.
        columns_to_decrypt (List[str]): A list of column names to be decrypted.

    Returns:
        pl.DataFrame: The DataFrame with specified columns decrypted.
    """
    log.info("Decrypting DataFrame in-memory...")
    
    if "canary" not in df.columns:
        raise ValueError("Decryption failed: 'canary' column not found in DataFrame.")

    # The canary column serves as the password for each row.
    password_column = "canary"

    # Define expressions for decrypting each specified column.
    # The `decrypt` function is applied to each element.
    decrypt_expressions = [
        pl.struct([col, password_column]).map_elements(
            lambda s: decrypt(s[col], s[password_column]),
            return_dtype=pl.String
        ).alias(col) for col in columns_to_decrypt
    ]

    # Apply the decryption expressions and then drop the canary column.
    decrypted_df = df.with_columns(decrypt_expressions).drop(password_column)
    
    log.info("In-memory decryption complete.")
    return decrypted_df 