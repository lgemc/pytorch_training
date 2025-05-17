import unittest

from q_and_a.prompts import prompt

class TestPrompt(unittest.TestCase):
    def test_prompt(self):
        question = "What is the capital of France?"
        options = ["Berlin", "Madrid", "Paris", "Rome"]
        augmented_items = ["France is a country in Europe.", "Paris is the capital of France."]

        result = prompt(question, options, augmented_items)
        self.assertIn(question, result)
        self.assertIn("Options:", result)
        self.assertIn("Answer:", result)
        self.assertIn("Context:", result)
        print(result)

        for item in augmented_items:
            self.assertIn(item, result)
        for option in options:
            self.assertIn(option, result)