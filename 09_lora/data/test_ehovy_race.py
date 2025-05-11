import unittest

from data.ehovy_race import EhovyRaceDataset

class TestEhovyRaceDataset(unittest.TestCase):
    def test_load(self):
        dataset = EhovyRaceDataset(variation="high", split="train")
        self.assertEqual(62445, len(dataset))

        first_question = dataset[0]
        self.assertEqual("C", first_question["answer"])
        self.assertEqual(['doctor', 'model', 'teacher', 'reporter'], first_question["options"])
        self.assertIn("We can know from the passage that the author works", first_question["question"])
        self.assertIn("Last week I talked with some of my students about what they wanted", first_question["article"])

    def test_load_filtered_by_size(self):
        max_size = 800
        dataset = EhovyRaceDataset(variation="high", split="train", max_article_size=max_size)

        self.assertEqual(803, len(dataset))