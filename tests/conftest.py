import pytest
import polars as pl
import os
import shutil

@pytest.fixture(scope="function")
def tmp_path(tmp_path_factory):
    """
    Creates a temporary directory for a test function.
    This is a wrapper around pytest's built-in tmp_path_factory to make cleanup explicit.
    """
    path = tmp_path_factory.mktemp("test_run_")
    yield path
    shutil.rmtree(path)

@pytest.fixture
def sample_plaintext_df():
    """Provides a sample Polars DataFrame with plaintext data."""
    data = {
        "secret_id": [101, 202, 303],
        "question": ["What is 1+1?", "Capital of France?", "Color of the sky?"],
        "secret_answer": ["2", "Paris", "Blue"],
        "useless_info": ["a", "b", "c"]
    }
    return pl.DataFrame(data)

@pytest.fixture
def sample_plaintext_csv(tmp_path, sample_plaintext_df):
    """Creates a sample plaintext CSV file and returns its path."""
    file_path = os.path.join(tmp_path, "plaintext.csv")
    sample_plaintext_df.write_csv(file_path)
    return file_path

# Fixtures for Pipeline Testing

@pytest.fixture
def sample_text_file(tmp_path):
    """Creates a sample text file for testing LocalFileSource."""
    content = "The first modern computer was the Z1, created by Konrad Zuse in 1936. The first commercial computer was the UNIVAC I, introduced in 1951."
    file_path = os.path.join(tmp_path, "test_document.txt")
    with open(file_path, 'w') as f:
        f.write(content)
    return file_path 