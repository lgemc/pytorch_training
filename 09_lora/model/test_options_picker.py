import unittest

import torch

from model.llama_3_2_tokenizer import Llama32Tokenizer
from model.llama_3_2 import Llama32
from model.options_picker import OptionsPicker

model_name = "meta-llama/Llama-3.2-1B"
class TestOptionsPicker(unittest.TestCase):
    def setUp(self):
        tokenizer = Llama32Tokenizer(model_name)
        model = Llama32(model_name, do_sample=False)
        self.options = ["A", "B", "C", "D"]
        self.options_picker = OptionsPicker(model, tokenizer, options=self.options, device="cuda")
    def test_options_picker(self):
        # Sample example
        with torch.no_grad():
            expected_prompt = f"""
                Context: This is a sample article.\n\n
                Questions: What is the main topic of the article?\n\n
                Options:\nA) Sample\nB) Example\nC) Test\nD) Demo\n\n
                Answer:
            """
            right_answer = "A"

            tokenized_prompt = self.options_picker.tokenizer(expected_prompt)
            out = self.options_picker(tokenized_prompt["input_ids"], tokenized_prompt["attention_mask"])
            answer = self.options[torch.argmax(out)]
            self.assertEqual(answer, right_answer)

    def test_get_option_ids(self):
        option_ids = self.options_picker._get_option_ids()
        self.assertEqual(option_ids, [32, 33, 34, 35])

    def test_model_generation(self):
        # Sample input
        with torch.no_grad():
            expected_prompt = f"""
                Context: This is a sample article.\n\n
                Questions: What is the main topic of the article?\n\n
                Options:\nA) Sample\nB) Example\nC) Test\nD) Demo\n\n
                Answer:
            """
            right_answer = "A"

            tokenized_prompt = self.options_picker.tokenizer(expected_prompt)
            out = self.options_picker.model.generate(tokenized_prompt["input_ids"], tokenized_prompt["attention_mask"], max_length=600)
            decoded_output = self.options_picker.tokenizer.decode(out[0])
            print(decoded_output)
