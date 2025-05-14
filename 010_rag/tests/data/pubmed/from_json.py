import unittest

from data.pubmed.from_json import FromJsonDataset

class TestFromJsonDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = FromJsonDataset(json_file="../../../data/pubmed_500K.json")

    def test_load_dataset(self):
        self.assertEqual(500_000, len(self.dataset))

    def test_print_first_item(self):
        first_item = self.dataset[0]
        self.assertIn('title', first_item)
        self.assertIn('content', first_item)
        self.assertIn('contents', first_item)
        self.assertIn('PMID', first_item)
        self.assertIn('id', first_item)