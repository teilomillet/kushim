import os
import pytest
import polars as pl
from dotenv import load_dotenv
import litellm

from kushim.pipeline import KushimPipeline
from kushim.config import KushimConfig
from kushim.source import LocalFileSource

# Load environment variables for the live test
load_dotenv()

# Determine if the test should be skipped
# This is a live test that makes real API calls. It should only run if an
# API key is available in the environment.
API_KEY_AVAILABLE = any(key.endswith('_API_KEY') for key in os.environ)

@pytest.fixture
def local_source(tmp_path):
    """Creates a temporary local file source for testing."""
    d = tmp_path / "test_data"
    d.mkdir()
    # Provide a slightly longer text to ensure there's enough content
    # for the optimization step to generate a few examples.
    p = d / "test_article.txt"
    p.write_text("Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC. The mission fulfilled a national goal set by President John F. Kennedy in 1961.")
    return LocalFileSource()

@pytest.mark.skipif(not API_KEY_AVAILABLE, reason="Requires API key for live testing.")
def test_pipeline_produces_high_quality_output(local_source, tmp_path):
    """
    Tests that the optimized pipeline produces high-quality, structured output.

    This is a high-level behavioral test that verifies the pipeline's end-goal:
    to generate a dataset of Q&A pairs that are not only structurally correct
    but also adhere to specific quality standards (e.g., conciseness),
    reflecting the logic in our advanced optimization metric.
    """
    config = KushimConfig(
        fetch_kwargs={"path": str(tmp_path / "test_data")},
        model_name='openrouter/openai/gpt-4.1',
        max_workers=1,
        question_style="simple",
    )

    pipeline = KushimPipeline(source=local_source, config=config)
    
    try:
        qa_dataset, source_docs = pipeline.run(optimize=True, num_optimization_examples=2)
    except litellm.BadRequestError as e:
        if "invalid_api_key" in str(e).lower():
            pytest.skip("Skipping test due to invalid API key in environment.")
        # Re-raise the exception if it's a different kind of bad request
        raise

    # 1. Structural Validation
    assert isinstance(qa_dataset, pl.DataFrame)
    assert not qa_dataset.is_empty(), "The pipeline should generate at least one valid Q&A pair."
    
    expected_columns = ["question", "answer", "source_chunk", "source_title", "source_metadata"]
    for col in expected_columns:
        assert col in qa_dataset.columns, f"DataFrame is missing expected column: {col}"
        
    # 2. Source Document Validation
    assert len(source_docs) == 1
    assert source_docs[0]['title'] == "test_article.txt"

    # 3. Content Quality Validation
    # This is the core of our high-level test. We are not just checking if the
    # code runs; we are checking if the output meets our desired quality
    # criteria, which in this case is defined by our `advanced_metric`.
    for row in qa_dataset.iter_rows(named=True):
        answer_word_count = len(row["answer"].split())
        assert 1 <= answer_word_count <= 5, f"Answer '{row['answer']}' is not concise (1-5 words)."

    print("\n" + "="*20)
    print("Behavioral Test Passed: Pipeline produced high-quality output.")
    print("="*20)
    print("Generated DataFrame:")
    print(qa_dataset)
    print("\nSource Documents:")
    print(source_docs) 