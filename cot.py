import dspy
from dotenv import load_dotenv

load_dotenv()

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=1000)
dspy.settings.configure(lm=turbo)


# DSPy code
class GenerateAnswer(dspy.Signature):
    """Generate answers to the questions"""

    context = dspy.InputField(desc="may contain relevant facts and psychological insights")
    question = dspy.InputField()
    answer_choices = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class CoTSimplifiedBaleen(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question, context, answer_choices):
        pred = self.generate_answer(context=context, question=question, answer_choices=answer_choices)
        return dspy.Prediction(context=context, answer=pred.answer)
    