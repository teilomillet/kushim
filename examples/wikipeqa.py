"""
This script provides a complete example of how to use the Kushim framework
to generate a high-quality, verifiable Q&A dataset from a Wikipedia article.

It demonstrates the simple, high-level pipeline function.
"""
import os
import json
import kushim
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the Q&A generation pipeline.
    """
    # model_to_use = "openai/gpt-4.1" # Example for OpenAI
    model_to_use = "openrouter/openai/gpt-4.1" # Default, fast and free model

    # To generate questions from a single, specific Wikipedia article,
    # simply provide the article title.
    # article_title = "Supermarine Spitfire"
    # logging.info("Starting Q&A dataset generation for article: '%s'", article_title)
    # validated_qa_dataset, source_articles = kushim.pipeline.generate_qa_dataset(
    #     article_title=article_title, model_name=model_to_use
    # )
    # output_filename_base = article_title.replace(' ', '_')
    
    
    # For a more powerful approach, you can provide a search query.
    # Kushim will find the most relevant articles, combine their content,
    # and generate questions from that broader knowledge base.
    search_query = "Saturn"
    logging.info("Starting Q&A dataset generation for query: '%s'", search_query)
    validated_qa_dataset, source_articles = kushim.pipeline.generate_qa_dataset(
        query=search_query, num_articles_from_query=8, model_name=model_to_use
    )
    output_filename_base = search_query.replace(' ', '_')


    # For advanced use, you can call the individual components yourself:
    #
    # chunks = kushim.pipeline.fetch_and_chunk_from_query(search_query)
    # raw_pairs = kushim.pipeline.generate_qa_pairs(chunks)
    # validated_qa_dataset = kushim.pipeline.validate_qa_pairs(raw_pairs)

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
        logging.warning("No validated Q&A pairs were generated for this query.")
        print("\\nNo validated Q&A pairs were generated for this query.")

if __name__ == "__main__":
    main() 