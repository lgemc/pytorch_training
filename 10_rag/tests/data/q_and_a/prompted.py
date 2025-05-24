import unittest

from data.q_and_a.train_and_eval import TrainAndEval
from data.q_and_a.eval_with_answers import EvalWithAnswers
from data.q_and_a.prompted import Prompted

class TestPrompted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None

    def test_prompter_dataset(self):
        eval_dataset = TrainAndEval("../../../data/pubmed_QA_eval.json")
        eval_with_answers = EvalWithAnswers(eval_dataset)
        prompted = Prompted(eval_with_answers, mock_prompter)

        eval_with_answer = eval_with_answers[0]
        text, answer = prompted[0]

        self.assertIn(eval_with_answer["question"], text)

        for option in eval_with_answer["options"]:
            self.assertIn(option, text)

        self.assertIn(answer, range(4))

        print(text, answer)

def mock_prompter(question, options, _):
            return f"Question: {question}, Options: {options}"