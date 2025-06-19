import dspy
from typing import Dict, Type

# By providing detailed instructions and examples in the signature's docstring,
# we guide the LLM to generate higher-quality, more consistent outputs.
# This is a key principle of the DSPy framework.

class NarrativeQA(dspy.Signature):
    """
    Generate a complex, narrative-style question and a concise, factual answer from the provided context.

    The question should be a multi-clue puzzle. It should weave together several facts from the context into a narrative or a descriptive puzzle.
    The goal is to create a question where the answer is not obvious and requires synthesizing multiple pieces of information.
    The question should be framed naturally, as if someone is trying to recall something by listing the details they remember.
    Avoid simple, direct questions. Do not start with phrases like "I remember...".

    The answer must be explicitly stated in the text and be very short (ideally 1-4 words).
    """
    context = dspy.InputField(desc="A passage of text to generate a question from.")
    question = dspy.OutputField(desc="A complex, narrative-style question that synthesizes multiple facts into a puzzle.")
    answer = dspy.OutputField(desc="A concise, 1-4 word factual answer to the question, found directly in the text.")

class SimpleFactQA(dspy.Signature):
    """
    Generate a simple, direct, factual question and a concise answer from the context.
    The question should be a straightforward query about a single fact.
    The answer must be explicitly stated in the text and be very short (ideally 1-5 words).
    """
    context = dspy.InputField(desc="A passage of text to generate a question from.")
    question = dspy.OutputField(desc="A simple, factual question about a single piece of information.")
    answer = dspy.OutputField(desc="A concise, factual answer to the question.")

class BooleanQA(dspy.Signature):
    """
    Generate a question that has a 'Yes' or 'No' answer based on the context.
    The answer must be either 'Yes' or 'No'.
    """
    context = dspy.InputField(desc="A passage of text to generate a 'Yes' or 'No' question from.")
    question = dspy.OutputField(desc="A question that can be answered with 'Yes' or 'No'.")
    answer = dspy.OutputField(desc="A 'Yes' or 'No' answer.")


# Question Style Registry
# This registry maps user-friendly style names to the corresponding dspy.Signature class.
# It allows the pipeline to be dynamically configured to produce different types of Q&A pairs.
QUESTION_STYLE_REGISTRY: Dict[str, Type[dspy.Signature]] = {
    "narrative": NarrativeQA,
    "simple": SimpleFactQA,
    "boolean": BooleanQA,
}


class QAGeneration(dspy.Module):
    """
    A DSPy module for generating question-answer pairs from a given context.
    This module is now style-agnostic. It takes a signature class as input
    and uses it to drive the generation process.
    """
    def __init__(self, signature: Type[dspy.Signature], num_questions_per_chunk=1):
        super().__init__()
        # The generation logic is now driven by the provided signature.
        self.generate_qa = dspy.ChainOfThought(signature, n=num_questions_per_chunk)

    def forward(self, context):
        """
        Generates a list of question-answer pairs for the given context
        based on the configured signature.
        """
        return self.generate_qa(context=context)
