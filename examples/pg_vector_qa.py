"""
This script provides a complete, end-to-end example of the Kushim framework.
It demonstrates a full data pipeline:
1. Sourcing data from an external API (Wikipedia).
2. Indexing that data into a vector database (PGVector).
3. Using the indexed data as a source for Q&A generation.
"""
import os
import json
import logging
from sqlalchemy import make_url
from dotenv import load_dotenv

from kushim import pipeline, config, source
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables from .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection details
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "kushim_db")
TABLE_NAME = os.getenv("DB_TABLE_NAME", "kushim_e2e_documents")

# Model for generation and validation
MODEL_TO_USE = "openrouter/openai/gpt-4.1"
# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE")
EMBEDDING_DIM = 3584

DATA_THEME = "renouvellement passeport" # A general theme for metadata filtering
SEARCH_QUERY = "Comment renouveler mon passeport en urgence ?" # A specific query for semantic search

def setup_database_and_get_store() -> PGVectorStore:
    """
    Sets up the database, table, and vector extension.
    Returns a PGVectorStore instance.
    """
    # Connection URL for the database
    connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    url = make_url(connection_string)
    logging.info(f"Connection string to db : {url}")

    # Initialize the vector store
    vector_store = PGVectorStore.from_params(
        database=DB_NAME,
        host=url.host,
        port=str(url.port),
        user=url.username,
        password=url.password,
        table_name=TABLE_NAME,
        embed_dim=EMBEDDING_DIM,
        hybrid_search=True # Recommended for combining keyword and vector search
    )
    return vector_store

def main():
    """
    Main function to orchestrate the Q&A generation pipeline from a vector DB.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please add it to your .env file or export it.")

    Settings.embed_model = OpenAIEmbedding(model_name=EMBEDDING_MODEL, api_base=EMBEDDING_API_BASE, api_key=OPENAI_API_KEY)
    logging.info(f"Embedding api base : {EMBEDDING_API_BASE}")

    logging.info("Setting up vector database")
    vector_store = setup_database_and_get_store()

    logging.info("Configuring Kushim Pipeline for retrieval")
    # Now, we use the VectorDBSource to pull the data we just indexed.
    data_source = source.VectorDBSource(vector_store=vector_store)
    
    fetch_kwargs = {"theme": DATA_THEME, "query": SEARCH_QUERY}
    output_filename_base = f"{DATA_THEME}_qa"

    # Define a file path to save/load the compiled (optimized) generator.
    compiled_program_path = f"compiled_{output_filename_base}_generator.json"

    # Create the configuration for the pipeline.
    pipeline_config = config.KushimConfig(
        model_name=MODEL_TO_USE,
        fetch_kwargs=fetch_kwargs,
        num_questions_per_chunk=3, # Ask for more questions per chunk
        max_workers=8,
    )

    # Instantiate the main Kushim pipeline.
    kushim_pipeline = pipeline.KushimPipeline(
        source=data_source,
        config=pipeline_config
    )
    
    logging.info(f"Running the pipeline for theme '{DATA_THEME}'")
    validated_qa_dataset, source_articles = kushim_pipeline.run(
        optimize=True,
        compiled_generator_path=compiled_program_path
    )

    if validated_qa_dataset is not None and not validated_qa_dataset.is_empty():
        print(f"\nSuccessfully generated {len(validated_qa_dataset)} validated Q&A pairs.")
        print(validated_qa_dataset)

        # Save the final dataset to a CSV file.
        output_dir = "datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        qa_output_path = os.path.join(output_dir, f"{output_filename_base}_dataset.csv")
        sources_output_path = os.path.join(output_dir, f"{output_filename_base}_sources.json")

        logging.info("Saving Q&A dataset to %s", qa_output_path)
        validated_qa_dataset.write_csv(qa_output_path)
        print(f"\nQ&A dataset saved to: {qa_output_path}")

        logging.info("Saving source articles to %s", sources_output_path)
        with open(sources_output_path, 'w', encoding='utf-8') as f:
            json.dump(source_articles, f, ensure_ascii=False, indent=4)
        print(f"Source articles saved to: {sources_output_path}")

    else:
        logging.warning("No validated Q&A pairs were generated.")
        print("\nNo validated Q&A pairs were generated.")

if __name__ == "__main__":
    main() 