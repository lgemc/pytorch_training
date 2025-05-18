import unittest

from data.q_and_a.train_and_eval import TrainAndEval
from data.q_and_a.eval_with_answers import EvalWithAnswers

from q_and_a.prompts import prompt

class TestPrompt(unittest.TestCase):
    def test_prompt(self):
        question = "What is the capital of France?"
        options = ["Berlin", "Madrid", "Paris", "Rome"]
        augmented_items = ["France is a country in Europe.", "Paris is the capital of France."]

        result = prompt(question, options, augmented_items)
        self.assertIn(question, result)
        self.assertIn("Options:", result)
        self.assertIn("Answer", result)
        self.assertIn("Context:", result)
        print(result)

        for item in augmented_items:
            self.assertIn(item, result)
        for option in options:
            self.assertIn(option, result)

    def test_generate_from_dataset(self):
        # Initialize the training and evaluation class
        eval_dataset = TrainAndEval("../../data/pubmed_QA_eval.json")
        eval_with_answers = EvalWithAnswers(eval_dataset)

        for i in range(200, 220):
            item = eval_with_answers[i]
            question = item["question"]
            options = item["options"]
            augmented_items = ["lets to it"]

            result = prompt(question, options, augmented_items)
            self.assertIn(question, result)
            self.assertIn("Options:", result)
            self.assertIn("Answer", result)
            self.assertIn("Context:", result)
            print(result)

            for item in augmented_items:
                self.assertIn(item, result)
            for option in options:
                self.assertIn(option, result)
