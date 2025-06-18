import polars as pl
import dspy
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm

# We are importing the BootstrapFewShotWithRandomSearch, which is a more
# advanced teleprompter. It will programmatically generate and evaluate
# few-shot prompts to find the one that works best for our specific task
# and data, creating a self-improving pipeline.
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from .source import Source, SourceDocument
from .chunking import chunk_text
from .qagen import QAGeneration, QUESTION_STYLE_REGISTRY
from .validation import QAValidationModule, QAValidation
from .config import KushimConfig

# This new metric function provides a more nuanced evaluation of the generated
# Q&A pairs. Instead of a simple boolean check, it scores the output based on
# multiple criteria: correctness and conciseness. This gives the DSPy optimizer
# a much richer signal to learn from, pushing the model to generate not just
# valid, but high-quality results. This approach is a best practice for
# advanced DSPy development.
def advanced_metric(validator: QAValidationModule):
    """
    Creates a metric function that validates correctness and conciseness.
    """
    def metric(ex, pred, trace=None):
        # 1. Correctness Check
        is_correct = validator(question=pred.question, answer=pred.answer, source_chunk=ex.context, return_bool=True)
        if not is_correct:
            return 0.0

        # 2. Conciseness Check
        # We reward answers that are short and to the point.
        answer_word_count = len(pred.answer.split())
        is_concise = 1 <= answer_word_count <= 5
        
        return float(is_concise)

    return metric

# The pipeline is now a class, which encapsulates the logic and configuration.
# This makes the system more modular, maintainable, and extensible.
class KushimPipeline:
    """
    The main Kushim pipeline for generating verifiable Q&A datasets.

    This class orchestrates the entire workflow. It now includes a self-
    optimization step, where it generates initial Q&A pairs, validates them
    to create a training set, and then uses a DSPy teleprompter to optimize
    the generation module before processing the full dataset.
    """
    def __init__(self, source: Source, config: KushimConfig):
        self.source = source
        self.config = config
        self._configure_dspy()
        # The QA validator and the metric are now initialized once to be reused.
        # This is more efficient and centralizes the evaluation logic.
        self.validator = QAValidationModule()
        self.metric = advanced_metric(self.validator)

    def _configure_dspy(self):
        """Configures the DSPy environment based on the pipeline's settings."""
        if dspy.settings.lm is None:
            if self.config.llm is None:
                try:
                    llm = dspy.LM(self.config.model_name)
                    dspy.configure(lm=llm)
                except Exception as e:
                    raise RuntimeError(f"Failed to instantiate dspy.LM with model '{self.config.model_name}'.") from e
            else:
                dspy.configure(lm=self.config.llm)

    def _fetch_and_chunk(self) -> Tuple[List[Tuple[str, SourceDocument]], List[SourceDocument]]:
        """Fetches documents, chunks them, and returns (chunk, doc) bundles."""
        documents = self.source.fetch(**self.config.fetch_kwargs)
        if not documents:
            print("No documents found.")
            return [], []

        chunk_bundles = []
        for doc in documents:
            chunks = chunk_text(doc['content'], chunk_size=self.config.chunk_size, overlap=self.config.chunk_overlap)
            for chunk in chunks:
                chunk_bundles.append((chunk, doc))
        
        print(f"Generated {len(chunk_bundles)} chunks from {len(documents)} documents.")
        return chunk_bundles, documents

    def _generate_qa_pairs_stream(self, chunk_bundles: List[Tuple[str, SourceDocument]], qa_generator: dspy.Module):
        """Generates Q&A pairs from chunks and yields them one by one."""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_bundle = {executor.submit(qa_generator, context=chunk): (chunk, doc) for chunk, doc in chunk_bundles}
            
            for i, future in enumerate(as_completed(future_to_bundle)):
                chunk, doc = future_to_bundle[future]
                print(f"Generating Q&A for chunk {i+1}/{len(chunk_bundles)}...")
                try:
                    prediction = future.result()
                    qa_list = prediction.qa_pairs if hasattr(prediction, 'qa_pairs') else [prediction]
                    for qa in qa_list:
                        yield {
                            "question": qa.question,
                            "answer": qa.answer,
                            "source_chunk": chunk,
                            "source_title": doc['title'],
                            "source_metadata": str(doc['metadata'])
                        }
                except litellm.BadRequestError:
                    # Re-raise critical infrastructure errors so the caller can handle them.
                    raise
                except Exception as e:
                    # For other errors, log them and attempt to continue processing.
                    print(f"Q&A generation for chunk from '{doc['title']}' failed: {e}")

    def _validate_qa_pairs_stream(self, qa_pair_stream) -> dict:
        """Validates a stream of Q&A pairs in parallel, yielding the valid ones."""

        def validate_row(row: dict):
            """Wraps validation logic for use with executor.map."""
            # This print statement is useful for monitoring progress in a stream.
            print(f"Validating: {row['question'][:60]}...")
            try:
                result = self.validator(question=row["question"], answer=row["answer"], source_chunk=row["source_chunk"])
                if result.is_valid:
                    return row
            except litellm.BadRequestError:
                # Re-raise critical infra errors so the caller can handle them.
                raise
            except Exception as e:
                print(f"Validation for question '{row['question']}' failed: {e}")
            return None

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # executor.map applies the function to each item from the input stream
            # in a parallel, streaming fashion. It's memory-efficient.
            validated_stream = executor.map(validate_row, qa_pair_stream)
            for validated_row in validated_stream:
                if validated_row:
                    yield validated_row

    def _create_training_set_from_data(self, chunk_bundles: List[Tuple[str, SourceDocument]], num_examples: int) -> List[dspy.Example]:
        """
        Generates a small, high-quality training set by generating, then validating,
        Q&A pairs from a sample of the source data. This is the core of the
        self-improvement loop.
        """
        print(f"Creating a training set from {num_examples} data samples...")
        initial_generator = self._get_base_generator()
        
        # Generate initial Q&A pairs from a subset of the data
        # We still use a batch operation here as the training set is small.
        initial_qa_stream = self._generate_qa_pairs_stream(chunk_bundles[:num_examples], initial_generator)
        initial_qa_df = pl.DataFrame(list(initial_qa_stream))
        if initial_qa_df.is_empty():
            return []

        # Validate these pairs to create labeled examples for the optimizer
        training_set = []
        for row in initial_qa_df.iter_rows(named=True):
            validation_result = self.validator(question=row["question"], answer=row["answer"], source_chunk=row["source_chunk"])
            example = dspy.Example(
                context=row["source_chunk"],
                question=row["question"],
                answer=row["answer"],
                is_valid=validation_result.is_valid
            ).with_inputs("context")
            training_set.append(example)
            
        print(f"Created a training set with {len(training_set)} examples.")
        return training_set

    def _get_base_generator(self) -> QAGeneration:
        """Returns a non-optimized Q&A generator module."""
        signature = QUESTION_STYLE_REGISTRY.get(self.config.question_style)
        if not signature:
            raise ValueError(f"Unknown question style: '{self.config.question_style}'.")
        return QAGeneration(signature=signature, num_questions_per_chunk=self.config.num_questions_per_chunk)

    def run(self, optimize: bool = True, num_optimization_examples: int = 10) -> Tuple[pl.DataFrame, List[SourceDocument]]:
        """
        Executes the end-to-end pipeline with an optional optimization step.

        Args:
            optimize: If True, runs the self-improvement loop to find a better
                      few-shot prompt before full generation.
            num_optimization_examples: The number of data chunks to use for building
                                     the optimization training set.

        Returns:
            A tuple containing the validated Q&A DataFrame and the list of source documents.
        """
        chunk_bundles, source_documents = self._fetch_and_chunk()
        if not chunk_bundles:
            return pl.DataFrame(), []

        if optimize:
            # 1. Create a training set programmatically from the source data
            trainset = self._create_training_set_from_data(chunk_bundles, num_examples=num_optimization_examples)
            
            if not trainset:
                print("Warning: Could not create a training set. Falling back to the base generator. This may happen if the generation model is failing or producing invalid formats.")
                optimized_generator = self._get_base_generator()
            else:
                # 2. The pipeline now uses the more sophisticated, multi-faceted metric
                #    to guide the optimization process.
                optimizer = BootstrapFewShotWithRandomSearch(
                    metric=self.metric,
                    max_bootstrapped_demos=2,
                    num_candidate_programs=4,
                )
                base_generator = self._get_base_generator()
                optimized_generator = optimizer.compile(base_generator, trainset=trainset)
                print("Optimization complete. Running full generation with the optimized module.")
        else:
            optimized_generator = self._get_base_generator()

        # 4. Run the full generation and validation process using a streaming pipeline
        qa_pair_stream = self._generate_qa_pairs_stream(chunk_bundles, optimized_generator)
        validated_qa_stream = self._validate_qa_pairs_stream(qa_pair_stream)
        
        # Collect the final results from the stream into a DataFrame
        validated_dataset = pl.DataFrame(list(validated_qa_stream))
        
        return validated_dataset, source_documents
