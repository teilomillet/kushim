import pytest
import polars as pl
import os
from kushim.pipeline import generate_qa_dataset
from kushim.source import LocalFileSource, WikipediaSource, SourceDocument

# A marker to skip tests that require a live API key if not available.
# To run these tests, a .env file with a valid API key (e.g., GROQ_API_KEY) must be present.
requires_api_key = pytest.mark.skipif(
    not os.path.exists(".env"), reason="requires .env file with API key"
)

def test_local_file_source(sample_text_file):
    """
    Tests that the LocalFileSource can correctly read a file and return
    it in the standardized SourceDocument format.
    """
    source = LocalFileSource()
    documents = source.fetch(path=sample_text_file)

    assert len(documents) == 1
    doc = documents[0]
    assert isinstance(doc, dict)
    assert doc['title'] == 'test_document.txt'
    assert 'Z1, created by Konrad Zuse' in doc['content']
    assert doc['metadata']['path'] == sample_text_file

@requires_api_key
def test_e2e_pipeline_with_local_source(sample_text_file):
    """
    Performs a live, end-to-end test of the `generate_qa_dataset` pipeline
    using a real LLM.

    This test verifies that the pipeline can:
    - Ingest data from a `LocalFileSource`.
    - Generate and validate Q&A pairs using a live `dspy` setup.
    - Return a non-empty, correctly structured DataFrame.
    """
    local_source = LocalFileSource()

    # Run the entire pipeline against a live model
    validated_dataset, source_docs = generate_qa_dataset(
        source=local_source,
        fetch_kwargs={'path': sample_text_file},
        question_style="simple"
    )

    # 1. Verify the source document was loaded
    assert len(source_docs) == 1
    assert source_docs[0]['title'] == 'test_document.txt'

    # 2. Verify the final dataset structure and content
    # With a live LLM, we can't assert specific content, but we can
    # check that the process completed and returned a valid, non-empty dataset.
    assert isinstance(validated_dataset, pl.DataFrame)
    assert not validated_dataset.is_empty()
    # Check for the new, enriched column structure
    expected_columns = ["question", "answer", "source_chunk", "source_title", "source_metadata"]
    assert all(col in validated_dataset.columns for col in expected_columns)
    
    # 3. Verify that the source content and metadata are correctly propagated
    first_row = validated_dataset.row(0, named=True)
    assert isinstance(first_row['source_chunk'], str)
    assert "Konrad Zuse" in first_row['source_chunk']
    
    # Check that the metadata from the original document is present
    assert first_row['source_title'] == 'test_document.txt'
    assert 'test_document.txt' in first_row['source_metadata'] 