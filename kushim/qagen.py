import dspy

# By providing detailed instructions and examples in the signature's docstring,
# we guide the LLM to generate higher-quality, more consistent outputs.
# This is a key principle of the DSPy framework.
class GenerateQA(dspy.Signature):
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


class QAGenerationModule(dspy.Module):
    """
    A DSPy Module for generating question-answer pairs.

    This module is designed to be "telepromptable," meaning its behavior can be
    optimized with a set of examples. By providing few-shot examples, we
    steer the language model to produce outputs that align with our specific
    requirements for atomicity and verifiability.
    """
    def __init__(self):
        super().__init__()
        # The teleprompter will compile the examples into a few-shot prompt.
        # This is more effective than simple zero-shot prompting.
        self.generate_qa = dspy.ChainOfThought(GenerateQA)

    def forward(self, context: str):
        """
        Executes the Q&A generation task.

        Args:
            context: The text chunk to process.

        Returns:
            A dspy.Prediction object containing the generated question and answer.
        """
        return self.generate_qa(context=context)


class QAGeneration(dspy.Module):
    """
    A DSPy module for generating question-answer pairs from a given context.
    """
    def __init__(self, num_questions_per_chunk=1):
        super().__init__()
        self.generate_qa = dspy.ChainOfThought(GenerateQA, n=num_questions_per_chunk)

    def forward(self, context):
        """
        Generates a list of question-answer pairs for the given context.
        """
        return self.generate_qa(context=context)


class GenerateAnswer(dspy.Signature):
    """
    Generate a question and a concise, few-word answer from a given context.
    The answer should be as short as possible, ideally 1-4 words.
    """
    context = dspy.InputField(desc="A passage of text to generate a question from.")
    question = dspy.OutputField(desc="A question that can be answered from the context.")
    answer = dspy.OutputField(desc="A concise, 1-4 word answer to the question.")
