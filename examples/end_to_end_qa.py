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
import psycopg2
from sqlalchemy import make_url
from dotenv import load_dotenv

from kushim import pipeline, config, source
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import Settings
from llama_index.embeddings.jinaai import JinaEmbedding

# Load environment variables from .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection details
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "kushim_db")
TABLE_NAME = "kushim_e2e_documents"

# Model for generation and validation
MODEL_TO_USE = "openrouter/openai/gpt-4.1"
# Embedding model for JinaAI.
# See https://jina.ai/embeddings/ for the latest models.
JINA_EMBEDDING_MODEL = "jina-embeddings-v4"
# The dimension for jina-embeddings-v4 is 2048, but the API has a limit of 1024.
# See: https://huggingface.co/jinaai/jina-embeddings-v4
EMBEDDING_DIM = 1024

WIKI_SEARCH_QUERY = "Artificial Intelligence"
WIKI_NUM_ARTICLES = 1
DATA_THEME = "AI" # The theme to assign to the indexed documents

def setup_database_and_get_store() -> PGVectorStore:
    """
    Sets up the database, table, and vector extension.
    Returns a PGVectorStore instance.
    """
    # Create the database if it doesn't exist
    conn_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"
    conn = psycopg2.connect(conn_string)
    conn.autocommit = True
    with conn.cursor() as c:
        c.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        if not c.fetchone():
            c.execute(f"CREATE DATABASE {DB_NAME}")
    conn.close()

    # Connection URL for the new database
    connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    url = make_url(connection_string)

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

def fetch_and_index_wikipedia_articles(vector_store: PGVectorStore, query: str, num_articles: int, theme: str):
    """
    Fetches articles from Wikipedia and indexes them into the vector store,
    skipping any articles that have already been indexed for the given theme.
    """
    existing_titles = set()
    try:
        # Construct a valid DSN for psycopg2
        conn_string = f"dbname='{DB_NAME}' user='{DB_USER}' host='{DB_HOST}' password='{DB_PASSWORD}' port='{DB_PORT}'"
        conn = psycopg2.connect(conn_string)
        conn.autocommit = True
        with conn.cursor() as c:
            # The metadata is in a JSONB column named `_node_metadata` by default
            c.execute(
                f"SELECT _node_metadata ->> 'title' FROM {TABLE_NAME} WHERE _node_metadata ->> 'theme' = %s",
                (theme,)
            )
            results = c.fetchall()
            existing_titles = {row[0] for row in results}
            if existing_titles:
                logging.info(f"Found {len(existing_titles)} existing articles for theme '{theme}'.")
    except psycopg2.Error as e:
        # PostgreSQL error code for 'undefined_table' is 42P01
        if e.pgcode == '42P01':
            logging.info("Table 'kushim_e2e_documents' not found. This is expected on the first run. Proceeding to create and index.")
        else:
            # Handle other potential database errors
            logging.warning(f"An unexpected database error occurred while checking for existing articles: {e}")
    finally:
        if 'conn' in locals() and not conn.closed:
            conn.close()

    logging.info(f"Fetching {num_articles} Wikipedia articles for query: '{query}'")
    wiki_source = source.WikipediaSource()
    articles = wiki_source.fetch(mode="search", query=query, num_articles_to_return=num_articles)

    documents_to_index = []
    for article in articles:
        if article['title'] not in existing_titles:
            logging.info(f"Preparing to index new article: '{article['title']}'")
            doc = Document(
                text=article['content'],
                metadata={
                    "title": article['title'],
                    "theme": theme,
                    "source": "wikipedia"
                }
            )
            documents_to_index.append(doc)
        else:
            logging.info(f"Skipping already indexed article: '{article['title']}'")

    if documents_to_index:
        logging.info(f"Indexing {len(documents_to_index)} new documents into the vector store...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            documents_to_index,
            storage_context=storage_context,
        )
        logging.info("Indexing complete.")
    else:
        logging.info("No new documents to index.")

def main():
    """
    Main function to orchestrate the Q&A generation pipeline from a vector DB.
    """
    # Set the embedding model for the LlamaIndex settings
    # The JinaEmbedding will automatically use the JINA_API_KEY environment variable if it is set.
    # Passing it explicitly is a more robust way to ensure it's used.
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY environment variable not set. Please get a key from https://jina.ai/embeddings/ and add it to your .env file.")
    
    Settings.embed_model = JinaEmbedding(
        model_name=JINA_EMBEDDING_MODEL,
        api_key=api_key,
        dimensions=EMBEDDING_DIM
    )

    logging.info("Setting up vector database")
    vector_store = setup_database_and_get_store()

    logging.info(f"Sourcing and indexing data for theme '{DATA_THEME}'")
    # This function will fetch articles from Wikipedia and load them into our DB.
    # In a real application, this might be a separate, scheduled script.
    fetch_and_index_wikipedia_articles(
        vector_store=vector_store,
        query=WIKI_SEARCH_QUERY,
        num_articles=WIKI_NUM_ARTICLES,
        theme=DATA_THEME
    )

    logging.info("Configuring Kushim Pipeline for retrieval")
    # Now, we use the VectorDBSource to pull the data we just indexed.
    data_source = source.VectorDBSource(vector_store=vector_store)
    
    fetch_kwargs = {"theme": DATA_THEME, "top_k": WIKI_NUM_ARTICLES}
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