import unittest

from data.ehovy_race import EhovyRaceDataset
from data.prompt_dataset import PromptedEhvoy, prompt_with_question, extract_answer

class TestPromptedEhvyDataset(unittest.TestCase):
    def test_prompt_with_question(self):
        # Sample example
        example = {
            'article': 'This is a sample article.',
            'question': 'What is the main topic of the article?',
            'options': ['Sample', 'Example', 'Test', 'Demo']
        }

        expected_prompt = (
            "Context: This is a sample article.\n\n"
            "Questions: What is the main topic of the article?\n\n"
            "Options:\nA) Sample\nB) Example\nC) Test\nD) Demo\n\n"
            "Answer:"
        )

        prompt = prompt_with_question(example)
        self.assertEqual(prompt, expected_prompt)

    def test_prompted_ehvy_dataset(self):
        # Initialize the dataset
        ehovy_dataset = EhovyRaceDataset(variation="high", split="test", max_article_size=800)
        prompted_dataset = PromptedEhvoy(ehovy_dataset, include_answer=True)

        # Check length
        self.assertEqual(len(prompted_dataset), len(ehovy_dataset))

        # Check a sample item
        sample_item, answer = prompted_dataset[0]
        self.assertIn("Context:", sample_item)
        self.assertIn("Question:", sample_item)
        self.assertIn("Options:", sample_item)
        self.assertIn("Answer:", sample_item)
        self.assertIsInstance(answer, str)
        self.assertIn(answer, ['A', 'B', 'C', 'D'])

        print(len(prompted_dataset))

    def test_extract_answer(self):
        # Test with a valid output
        output = "Answer: A"
        answer = extract_answer(output)
        self.assertEqual(answer, 'A')

        # Test with an invalid output
        output = "Answer: Z"
        answer = extract_answer(output)
        self.assertEqual(answer, 'Z')

        # Test with an empty output
        output = ""
        answer = extract_answer(output)
        self.assertIsNone(answer)

    def test_with_include_answer(self):
        # Sample example
        example = {
            'article': 'This is a sample article.',
            'question': 'What is the main topic of the article?',
            'options': ['Sample', 'Example', 'Test', 'Demo'],
            'answer': 'A'
        }

        expected_prompt = (
            "Context: This is a sample article.\n\n"
            "Question: What is the main topic of the article?\n\n"
            "Options:\nA) Sample\nB) Example\nC) Test\nD) Demo\n\n"
            "Answer: A"  # The model is expected to fill this part
        )

        prompt = prompt_with_question(example, include_answer=True)
        self.assertEqual(prompt, expected_prompt)

