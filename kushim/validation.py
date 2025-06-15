import dspy

# This signature defines the task for our LLM Judge. The detailed
# instructions in the docstring are crucial for guiding the LLM to
# perform a nuanced validation based on our specific criteria.
class QAValidation(dspy.Signature):
    """
    Given a question, an answer, and a context, verify if the answer is
    correct, relevant, and concise (ideally 1-4 words).
    """
    question = dspy.InputField(desc="The question to validate.")
    answer = dspy.InputField(desc="The answer to validate.")
    context = dspy.InputField(desc="The context to validate against.")
    is_valid = dspy.OutputField(
        desc="A boolean indicating if the answer is correct, relevant, and concise (1-4 words).",
    )


class QAValidationModule(dspy.Module):
    """
    A DSPy module for validating question-answer pairs against a source chunk.
    """
    def __init__(self):
        super().__init__()
        # The teleprompter will use provided examples to create a few-shot prompt,
        # making the validation much more robust than a zero-shot approach.
        self.validate_qa = dspy.ChainOfThought(QAValidation)

    def forward(self, question: str, answer: str, source_chunk: str) -> dspy.Prediction:
        """
        Validates a single question-answer pair.
        """
        # The module returns a dspy.Prediction object, which we can parse.
        result = self.validate_qa(
            question=question,
            answer=answer,
            context=source_chunk
        )
        # We need to access the `is_valid` attribute from the prediction.
        is_valid = result.is_valid
        return dspy.Prediction(is_valid=is_valid)
