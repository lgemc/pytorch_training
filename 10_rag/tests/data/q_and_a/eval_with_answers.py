from unittest import TestCase

from data.q_and_a.train_and_eval import TrainAndEval
from data.q_and_a.eval_with_answers import EvalWithAnswers

class TestEvalWithAnswers(TestCase):
    def test_eval_with_answers(self):
        # Initialize the training and evaluation class
        eval_dataset = TrainAndEval("../../../data/pubmed_QA_eval.json")
        eval_with_answers = EvalWithAnswers(eval_dataset)

        for i in range(10):
            # Get the question and answer
            item = eval_with_answers[i]
            statement_idx = item["options"].index(item["statement"])
            self.assertEqual(statement_idx, item["answer_idx"])