"""
This script provides a complete example of how to use the Kushim framework
to generate a high-quality, verifiable Q&A dataset from various sources.

It demonstrates the direct use of the core Kushim components, which is the
recommended approach for users of the library.
"""
import os
import json
import logging

from kushim import pipeline
from kushim import config
from kushim import source

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the Q&A generation pipeline.
    """
    # Define the model to use for generation and validation.
    # Make sure you have the necessary API keys set up in your environment
    # (e.g., OPENAI_API_KEY for OpenAI models).
    model_to_use = "openrouter/openai/gpt-4.1"
    
    # Initialize the data source. You can choose from different sources.
    # In this example, we use WikipediaSource to find articles by a search query.
    data_source = source.WikipediaSource()
    fetch_kwargs = {
        "mode": "search",
        "query": "Moon",
        "num_articles_to_return": 5,
    }
    output_filename_base = "moon"

    # Alternative Sources (Examples)
    # 1. Use a single Wikipedia article
    # data_source = source.WikipediaSource()
    # fetch_kwargs = {"mode": "article", "article_title": "Supermarine Spitfire"}
    # output_filename_base = "supermarine_spitfire"

    # 2. Use local text files from a directory
    # os.makedirs("my_documents", exist_ok=True)
    # with open("my_documents/doc1.txt", "w") as f:
    #     f.write("The first manned mission to Mars is planned for the 2030s.")
    # data_source = source.LocalFileSource()
    # fetch_kwargs = {"path": "my_documents"}
    # output_filename_base = "local_docs"

    # Define a file path to save/load the compiled (optimized) generator.
    # This avoids re-running the expensive optimization process on every execution.
    compiled_program_path = f"compiled_{output_filename_base}_generator.json"

    logging.info(f"Starting Q&A dataset generation for source with args: {fetch_kwargs}")

    # Create the configuration for the pipeline.
    # This includes the model name and the arguments for the data source.
    pipeline_config = config.KushimConfig(
        model_name=model_to_use,
        fetch_kwargs=fetch_kwargs,
        num_questions_per_chunk=2, # Generate 2 questions per chunk
        max_workers=8 # Use 8 parallel workers for generation/validation
    )

    # Instantiate the main Kushim pipeline with the source and config.
    kushim_pipeline = pipeline.KushimPipeline(
        source=data_source,
        config=pipeline_config
    )

    # Execute the pipeline.
    # If a compiled generator exists at `compiled_program_path`, it will be
    # loaded, and the optimization step will be skipped. Otherwise, the
    # pipeline will run optimization and save the result to the path.
    validated_qa_dataset, source_articles = kushim_pipeline.run(
        optimize=True,
        compiled_generator_path=compiled_program_path
    )

    if validated_qa_dataset is not None and not validated_qa_dataset.is_empty():
        print(f"\\nSuccessfully generated {len(validated_qa_dataset)} validated Q&A pairs.")
        print(validated_qa_dataset)

        # Save the final dataset to a CSV file.
        output_dir = "datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        # Define paths for the Q&A dataset and the sources file
        qa_output_path = os.path.join(output_dir, f"{output_filename_base}_qa_dataset.csv")
        sources_output_path = os.path.join(output_dir, f"{output_filename_base}_sources.json")

        # Save the Q&A dataset
        logging.info("Saving Q&A dataset to %s", qa_output_path)
        validated_qa_dataset.write_csv(qa_output_path)
        print(f"\\nQ&A dataset saved to: {qa_output_path}")

        # Save the source articles for reproducibility and further analysis
        logging.info("Saving source articles to %s", sources_output_path)
        with open(sources_output_path, 'w', encoding='utf-8') as f:
            json.dump(source_articles, f, ensure_ascii=False, indent=4)
        print(f"Source articles saved to: {sources_output_path}")

    else:
        logging.warning("No validated Q&A pairs were generated.")
        print("\\nNo validated Q&A pairs were generated.")

if __name__ == "__main__":
    main() 