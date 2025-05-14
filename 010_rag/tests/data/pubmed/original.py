import unittest

from src.data.pub_med import PubMed

class TestPubMED(unittest.TestCase):
    def setUp(self):
        self.dataset = PubMed(split="train")

    def test_load_dataset(self):
        self.assertEqual(23898701, len(self.dataset))

    def test_print_first_item(self):
        first_item = self.dataset[0]
        print(first_item)
        self.assertIn('title', first_item)
        self.assertIn('content', first_item)
        self.assertIn('contents', first_item)
        self.assertIn('PMID', first_item)
        self.assertIn('id', first_item)
