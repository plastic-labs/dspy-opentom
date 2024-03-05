import dspy


# DSPy code
class GenerateAnswer(dspy.Signature):
    """Generate answers to the questions"""

    context = dspy.InputField(desc="may contain relevant facts and psychological insights")
    question = dspy.InputField()
    thought = dspy.InputField(desc="a thought that might help answer the question")
    answer_choices = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateThought(dspy.Signature):
    """Generate thoughts about questions"""

    context = dspy.InputField(desc="may contain relevant facts and psychological insights")
    question = dspy.InputField()
    thought = dspy.OutputField(desc="a thought that might help answer the question")


class CoTWithThoughtSimplifiedBaleen(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_thought = dspy.ChainOfThought(GenerateThought)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question, context, answer_choices):
        pred_thought = self.generate_thought(context=context, question=question)
        pred = self.generate_answer(
            context=context, question=question, thought=pred_thought.thought, answer_choices=answer_choices
        )
        return dspy.Prediction(context=context, answer=pred.answer)
