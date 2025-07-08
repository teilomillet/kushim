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
# Set higher logging level for libraries to reduce noise, and debug for our script.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger(__name__).setLevel(logging.DEBUG)

# Environment Variable Validation
# We explicitly check for required environment variables to provide clear,
# immediate feedback if the environment is not configured correctly.
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "EMBEDDING_MODEL"]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise ValueError(f"Required environment variable '{var}' is not set. Please add it to your .env file or export it.")

# Database Connection Details
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "kushim_db")
TABLE_NAME = os.getenv("DB_TABLE_NAME", "kushim_e2e_documents")

# Model Configuration
MODEL_TO_USE = "openrouter/openai/gpt-4.1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE")

EMBEDDING_DIM = 3584

# Search Configuration
# This is the question we want to find answers for in our vector database.
SEARCH_QUERY = "Comment renouveler mon passeport en urgence ?" # A specific query for semantic search

def connect_to_existing_vector_store() -> PGVectorStore:
    """Return a **read-only** PGVectorStore connected to the *existing* table.

    Important implementation details:
    1. We set ``perform_setup=False`` so that **no DDL statements** are issued –
       this prevents unwanted schema changes that could hide the real data
       during debugging.
    2. ``debug=True`` enables SQLAlchemy echo which prints every SQL query. That
       gives immediate feedback that the similarity query is *actually* hitting
       Postgres.
    3. We do **not** pass ``hnsw_kwargs`` or any other tuning parameters. A
       plain cosine-distance query is enough for a first sanity-check.
    """
    connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    url = make_url(connection_string)

    logging.info(
        "Connecting to existing PGVector table '%s' (DB: %s on %s:%s)…",
        TABLE_NAME,
        url.database,
        url.host,
        url.port,
    )

    return PGVectorStore.from_params(
        database=DB_NAME,
        host=url.host,
        port=str(url.port),
        user=url.username,
        password=url.password,
        table_name=TABLE_NAME,
        embed_dim=EMBEDDING_DIM,
        # This is the "read-only" switch. It prevents the script from trying
        # to create or alter database tables.
        perform_setup=False,
        # This is the "show me the SQL" switch. It prints all database commands.
        debug=True,
        # We keep the search simple (dense vector search only) for this example.
        hybrid_search=False,
    )

def main():
    """
    Main function to orchestrate the Q&A generation pipeline from a vector DB.
    """
    # Here, we tell our program which AI model to use to convert our search
    # query into a vector (that list of numbers).
    Settings.embed_model = OpenAIEmbedding(model_name=EMBEDDING_MODEL, api_base=EMBEDDING_API_BASE, api_key=OPENAI_API_KEY)
    logging.debug(f"Using embedding model: '{EMBEDDING_MODEL}' from API base: {EMBEDDING_API_BASE or 'default'}")

    logging.info("Step 1: Connecting to the existing Vector Database")
    vector_store = connect_to_existing_vector_store()
    logging.info("Vector database connection ready.")

    # ------------------------------------------------------------------
    # Sanity-check: ensure embedding model dimension matches DB schema
    # ------------------------------------------------------------------
    # This is a critical safety check. We create a sample embedding and verify
    # its length (dimension). If it doesn't match what the database expects
    # (EMBEDDING_DIM), we stop the script with an error. This prevents silent
    # failures where the database returns no results simply because the vector
    # sizes are incompatible.
    sample_embedding = Settings.embed_model.get_query_embedding("dimension-check")
    if len(sample_embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch – model produced {len(sample_embedding)} dims "
            f"but the database expects {EMBEDDING_DIM}. Update EMBEDDING_DIM or switch model."
        )

    logging.debug("Embedding dimension validated (dims=%s)", len(sample_embedding))

    logging.info("Step 2: Configuring Kushim Pipeline for Retrieval")
    # The VectorDBSource is a Kushim component that knows how to fetch data
    # from the vector database we just connected to.
    data_source = source.VectorDBSource(vector_store=vector_store)
    
    # Keep the query kwargs minimal for first-step debugging – we just want
    # to know if *anything* comes back.
    vector_store_query_kwargs = {}
    
    # The `fetch_kwargs` dictionary tells the pipeline exactly what to search for.
    # In this case, it's our search query.
    fetch_kwargs = {"query": SEARCH_QUERY, "vector_store_kwargs": vector_store_query_kwargs}
    logging.debug("Pipeline configured with fetch_kwargs: %s", fetch_kwargs)

    # Quick interactive debug: print the raw matches before running the full
    # pipeline so that we know the DB is returning data.
    def _debug_simple_query():
        """
        Run a quick, simple search to confirm the connection works and we are
        getting relevant results back from the database.
        """
        from llama_index.core.vector_stores import VectorStoreQuery

        # 1. Convert our text query into a vector.
        query_emb = Settings.embed_model.get_query_embedding(SEARCH_QUERY)
        # 2. Ask the database for the 3 most similar vectors.
        res = vector_store.query(
            VectorStoreQuery(query_embedding=query_emb, similarity_top_k=3)
        )

        num_nodes = len(res.nodes) if res.nodes else 0
        logging.info("Debug query returned %d nodes", num_nodes)

        # 3. Print the results to the console.
        if res.nodes:
            for i, node in enumerate(res.nodes, 1):
                title = node.metadata.get("title", "Untitled")
                similarity = res.similarities[i-1] if res.similarities else 0.0
                logging.info("%d. %s (similarity=%.3f)", i, title, similarity)

    # We call our debug function here to perform the quick check.
    _debug_simple_query()

    # Define a file path to save/load the compiled (optimized) generator.
    # This is a performance optimization for more advanced use cases.
    compiled_program_path = f"compiled_vector_search_qa_generator.json"

    # Create the configuration for the pipeline.
    # This bundles up all the settings for the main Q&A generation task.
    pipeline_config = config.KushimConfig(
        model_name=MODEL_TO_USE,
        fetch_kwargs=fetch_kwargs,
        num_questions_per_chunk=3, # Ask for more questions per chunk
        max_workers=8,
    )

    # Instantiate the main Kushim pipeline.
    # This creates the main orchestrator object that will manage the whole process.
    kushim_pipeline = pipeline.KushimPipeline(
        source=data_source,
        config=pipeline_config
    )
    logging.info("Kushim pipeline configured.")
    
    logging.info(f"--- Step 3: Running Pipeline with Query: '{SEARCH_QUERY}' ---")
    # This is the main event. We kick off the process to:
    # 1. Fetch the most relevant documents from our vector database.
    # 2. Break them into smaller chunks.
    # 3. Use a powerful AI model (like GPT-4) to generate questions and answers
    #    from each chunk.
    # 4. Validate the generated Q&A pairs to ensure they are high quality.
    validated_qa_dataset, source_articles = kushim_pipeline.run(
        optimize=True,
        compiled_generator_path=compiled_program_path
    )

    if validated_qa_dataset is not None and not validated_qa_dataset.is_empty():
        print(f"\n✅ Successfully generated {len(validated_qa_dataset)} validated Q&A pairs.")
        print("--- Generated Q&A Dataset ---")
        print(validated_qa_dataset)
        print("-----------------------------")

        # Save the final dataset to a CSV file.
        # If we got good results, we save them to disk for later use.
        output_dir = "datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        qa_output_path = os.path.join(output_dir, "vector_search_qa_dataset.csv")
        sources_output_path = os.path.join(output_dir, "vector_search_qa_sources.json")

        logging.info("Saving Q&A dataset to %s", qa_output_path)
        validated_qa_dataset.write_csv(qa_output_path)
        print(f"\nQ&A dataset saved to: {qa_output_path}")

        logging.info("Saving source articles to %s", sources_output_path)
        with open(sources_output_path, 'w', encoding='utf-8') as f:
            json.dump(source_articles, f, ensure_ascii=False, indent=4)
        print(f"Source articles saved to: {sources_output_path}")

    else:
        logging.warning("Pipeline run completed, but no validated Q&A pairs were generated.")
        print("\n⚠️ No validated Q&A pairs were generated. Check the debug logs for potential issues.")

if __name__ == "__main__":
    main() 