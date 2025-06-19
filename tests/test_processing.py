import pytest
import os
import asyncio
import polars as pl
from polars.testing import assert_frame_equal
from aiohttp import web
from kushim.utils import encrypt_file, decrypt_file
from kushim.utils.processing import CANARY_PREFIX, _get_input_size

# Individual tests are now marked, removing the global marker.

@pytest.mark.asyncio
async def test_in_memory_encryption_decryption_e2e(sample_plaintext_csv, sample_plaintext_df, tmp_path):
    """
    Tests the full end-to-end encryption and decryption flow using the
    fast in-memory path. This simulates the user journey for small files.
    """
    encrypted_path = os.path.join(tmp_path, "encrypted.csv")
    decrypted_path = os.path.join(tmp_path, "decrypted.csv")
    
    # 1. Encrypt the file
    await encrypt_file(
        input_path=sample_plaintext_csv,
        output_path=encrypted_path,
        columns_to_encrypt=["secret_id", "question", "secret_answer"],
        canary_name="test_dataset"
    )
    
    # Verification for Encryption
    assert os.path.exists(encrypted_path)
    encrypted_df = pl.read_csv(encrypted_path)
    
    # Check that a 'canary' column was added and is consistent
    assert "canary" in encrypted_df.columns
    assert encrypted_df["canary"].n_unique() == 1
    assert encrypted_df["canary"][0].startswith(f"{CANARY_PREFIX} test_dataset:")
    
    # Check that the specified columns are encrypted (not equal to original)
    assert not encrypted_df["question"].equals(sample_plaintext_df["question"])
    assert not encrypted_df["secret_answer"].equals(sample_plaintext_df["secret_answer"])
    assert not encrypted_df["secret_id"].equals(sample_plaintext_df["secret_id"])
    
    # Check that non-encrypted columns are untouched
    assert_frame_equal(encrypted_df[["useless_info"]], sample_plaintext_df[["useless_info"]])

    # 2. Decrypt the file, requesting to drop the password column for a clean output
    await decrypt_file(
        input_path=encrypted_path,
        output_path=decrypted_path,
        password_column="canary",
        columns_to_decrypt=["secret_id", "question", "secret_answer"],
        drop_password_column=True
    )
    
    # Verification for Decryption
    assert os.path.exists(decrypted_path)
    # The decryption process outputs strings. To prevent Polars from
    # re-inferring numeric types from columns that look like numbers (e.g., 'secret_id'),
    # we set `infer_schema_length=0`. This forces all columns to be read as strings,
    # ensuring a correct comparison with the expected string-typed data.
    decrypted_df = pl.read_csv(decrypted_path, infer_schema_length=0)
    
    # After decryption, all columns that were processed will be strings.
    # We must cast the original dataframe's columns to string for a valid comparison.
    expected_df = sample_plaintext_df.with_columns(
        pl.col("secret_id").cast(pl.String),
        pl.col("question").cast(pl.String),
        pl.col("secret_answer").cast(pl.String)
    )

    # Check that the decrypted data matches the original source data
    # The 'useless_info' column was not processed, so it should retain its original type
    assert_frame_equal(decrypted_df.select(["secret_id", "question", "secret_answer"]), expected_df.select(["secret_id", "question", "secret_answer"]))
    assert_frame_equal(decrypted_df.select("useless_info"), sample_plaintext_df.select("useless_info"))

@pytest.mark.asyncio
async def test_chunked_encryption_decryption_e2e(sample_plaintext_csv, sample_plaintext_df, tmp_path):
    """
    Tests the full end-to-end flow using the robust chunked-processing path
    by forcing it with a very small size threshold.
    """
    encrypted_path = os.path.join(tmp_path, "encrypted_chunked.csv")
    decrypted_path = os.path.join(tmp_path, "decrypted_chunked.csv")

    # 1. Encrypt the file using the chunked path
    await encrypt_file(
        input_path=sample_plaintext_csv,
        output_path=encrypted_path,
        columns_to_encrypt=["secret_id", "question", "secret_answer"],
        canary_name="chunked_test",
        # Force chunked processing by setting threshold to 0 bytes
        size_threshold_gb=0 
    )

    # 2. Decrypt the file using the chunked path, requesting to drop the password column
    await decrypt_file(
        input_path=encrypted_path,
        output_path=decrypted_path,
        password_column="canary",
        columns_to_decrypt=["secret_id", "question", "secret_answer"],
        size_threshold_gb=0,
        drop_password_column=True
    )
    
    # Verification
    # We use `infer_schema_length=0` to ensure that columns containing numbers
    # are read as strings, matching the state of the data after decryption.
    decrypted_df = pl.read_csv(decrypted_path, infer_schema_length=0)
    
    # Since all columns are now strings, we must cast the original dataframe's
    # columns for a valid comparison.
    expected_df = sample_plaintext_df.with_columns(
        pl.all().cast(pl.String)
    )
    assert_frame_equal(decrypted_df, expected_df, check_column_order=False)

@pytest.mark.asyncio
async def test_smart_defaults_for_all_columns(sample_plaintext_csv, sample_plaintext_df, tmp_path):
    """
    Tests that if `columns_to_encrypt` is not provided, all columns are
    processed by default, and a canary_name is auto-derived.
    """
    encrypted_path = os.path.join(tmp_path, "encrypted_all.csv")
    decrypted_path = os.path.join(tmp_path, "decrypted_all.csv")

    # Encrypt without specifying columns or a canary_name
    await encrypt_file(
        input_path=sample_plaintext_csv,
        output_path=encrypted_path,
    )
    
    # Decrypt without specifying columns
    await decrypt_file(
        input_path=encrypted_path,
        output_path=decrypted_path,
        password_column="canary",
        drop_password_column=True
    )

    # Verification
    # We use `infer_schema_length=0` to ensure that columns containing numbers
    # are read as strings, matching the state of the data after decryption.
    decrypted_df = pl.read_csv(decrypted_path, infer_schema_length=0)
    
    # The decrypted frame should match the original plaintext (with all columns cast to string)
    expected_df = sample_plaintext_df.with_columns(
        pl.all().cast(pl.String)
    )
    assert_frame_equal(decrypted_df, expected_df, check_column_order=False)

@pytest.mark.asyncio
async def test_decryption_keeps_password_column_by_default(sample_plaintext_csv, tmp_path):
    """
    Tests that the decryption process KEEPS the canary column by default,
    which is the safer, non-destructive option.
    """
    encrypted_path = os.path.join(tmp_path, "encrypted_for_keep.csv")
    decrypted_path = os.path.join(tmp_path, "decrypted_with_keep.csv")

    # 1. Encrypt the file
    await encrypt_file(
        input_path=sample_plaintext_csv,
        output_path=encrypted_path,
        canary_name="keep_test"
    )

    # 2. Decrypt the file using default settings
    await decrypt_file(
        input_path=encrypted_path,
        output_path=decrypted_path,
        password_column="canary",
    )

    # Verification
    decrypted_df = pl.read_csv(decrypted_path)
    # Check that the 'canary' column is present in the output
    assert "canary" in decrypted_df.columns

@pytest.mark.asyncio
async def test_encryption_without_canary_prefix(sample_plaintext_csv, tmp_path):
    """
    Tests that `include_canary_prefix=False` correctly generates a canary
    without the long text prefix.
    """
    encrypted_path = os.path.join(tmp_path, "encrypted_no_prefix.csv")

    await encrypt_file(
        input_path=sample_plaintext_csv,
        output_path=encrypted_path,
        canary_name="no_prefix_test",
        include_canary_prefix=False
    )
    
    encrypted_df = pl.read_csv(encrypted_path)
    canary_value = encrypted_df["canary"][0]

    assert not canary_value.startswith(CANARY_PREFIX)
    assert canary_value.startswith("no_prefix_test:")

@pytest.mark.asyncio
async def test_default_output_path(sample_plaintext_csv, tmp_path):
    """
    Tests that if `output_path` is not provided, a default one is created
    in the current working directory.
    """
    # We need to change the CWD for this test to be reliable
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        await encrypt_file(
            input_path=sample_plaintext_csv,
            # No output_path specified
        )
        
        # Check that the default output file was created
        expected_output_filename = "plaintext.encrypted.csv"
        assert os.path.exists(expected_output_filename)
    finally:
        # Always change back to the original directory
        os.chdir(original_cwd)

@pytest.mark.asyncio
async def test_decrypt_file_autodetect_columns_mixed_encryption(tmp_path):
    """
    Tests that decrypt_file can run without a `columns_to_decrypt` list,
    and that it correctly decrypts the columns that are encrypted while
    leaving the unencrypted columns untouched.
    """
    # 1. Create a dataset with mixed encrypted/unencrypted data
    input_csv = tmp_path / "mixed_data.csv"
    original_df = pl.DataFrame({
        "id": [1, 2, 3],
        "important_data": ["secret1", "secret2", "secret3"],
        "unimportant_data": ["infoA", "infoB", "infoC"],
        "numeric_data": [100, 200, 300]
    })
    original_df.write_csv(input_csv)

    # 2. Encrypt only the 'important_data' column
    encrypted_path = await encrypt_file(
        input_path=str(input_csv),
        output_path=tmp_path / "mixed_encrypted.csv",
        columns_to_encrypt=["important_data"],
        canary_name="mixed-test"
    )

    # 3. Decrypt the file WITHOUT specifying columns
    # The system should try to decrypt all columns and gracefully handle failures.
    decrypted_path = await decrypt_file(
        input_path=encrypted_path,
        password_column="canary",
        output_path=tmp_path / "mixed_decrypted.csv",
        columns_to_decrypt=None, # Explicitly test the auto-detection
    )

    # 4. Assertions
    assert os.path.exists(decrypted_path)
    # When reading the CSV, polars will correctly infer the types. 'important_data'
    # was decrypted and is a string. 'unimportant_data' was never touched and is a string.
    # 'id' and 'numeric_data' were never touched and are integers.
    result_df = pl.read_csv(decrypted_path)

    # The decrypted column should match the original
    assert result_df["important_data"].to_list() == original_df["important_data"].to_list()

    # The other columns should be untouched and have their original types
    assert result_df["unimportant_data"].to_list() == original_df["unimportant_data"].to_list()
    assert result_df["numeric_data"].to_list() == original_df["numeric_data"].to_list()
    assert result_df["id"].to_list() == original_df["id"].to_list()

    # The canary column should still be present by default
    assert "canary" in result_df.columns


def test_get_input_size_local(tmp_path):
    """Test getting size of a local file."""
    p = tmp_path / "test.txt"
    p.write_text("hello")
    assert _get_input_size(str(p)) == 5


@pytest.mark.asyncio
async def test_get_input_size_remote(aiohttp_server):
    """Test getting size of a remote file."""
    app = web.Application()
    async def handler(request):
        # A proper response should include Content-Length for the size
        # detection to work. We set it explicitly here.
        body = b"hello world"
        return web.Response(body=body, headers={'Content-Length': str(len(body))})
    app.router.add_get("/", handler)
    server = await aiohttp_server(app)
    
    url = str(server.make_url("/"))
    
    # The `_get_input_size` function contains a blocking network call
    # (`urllib.request.urlopen`). Calling it directly from an async test
    # function would block the event loop and cause a deadlock.
    # To fix this, we run the blocking function in a separate thread using
    # `asyncio.to_thread`, which is the standard way to integrate blocking
    # I/O with asyncio.
    size = await asyncio.to_thread(_get_input_size, url)

    # Now we can assert the size is correct.
    assert size == 11 