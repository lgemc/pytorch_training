import unittest
import os
import torch.cuda
from huggingface_hub import login
login("<<token here>>")

from data.ehovy_race import EhovyRaceDataset
from data.prompt_dataset import PromptedEhvoy, extract_answer
from model.llama_3_2 import Llama32
from model.llama_3_2_tokenizer import Llama32Tokenizer

model_1B = "meta-llama/Llama-3.2-1B"

print(torch.cuda.is_available())

class TestLlama32Model(unittest.TestCase):
    def setUp(self):
        # Initialize the tokenizer and model
        self.tokenizer = Llama32Tokenizer(model_1B)
        self.model = Llama32(model_1B, do_sample=False)

        # Initialize the dataset
        self.ehovy_dataset = EhovyRaceDataset(variation="high", split="train")
        self.prompted_dataset = PromptedEhvoy(self.ehovy_dataset)

    def test_predict(self):
        x, y = self.prompted_dataset[0]
        tokenized_prompt = self.tokenizer(x)
        out = self.model.generate(
            input_ids=tokenized_prompt["input_ids"],
            attention_mask=tokenized_prompt["attention_mask"],
            max_length=600,
        )

        # Decode the generated output
        decoded_output = self.tokenizer.decode(out[0])
        answer = extract_answer(decoded_output)
        print(answer, y)

    def test_accuracy(self):
        correct_predictions = 0
        total_predictions = len(self.prompted_dataset)

        for i in range(total_predictions):
            x, y = self.prompted_dataset[i]
            tokenized_prompt = self.tokenizer(x)
            out = self.model.generate(
                input_ids=tokenized_prompt["input_ids"],
                attention_mask=tokenized_prompt["attention_mask"],
                max_length=900,
            )

            # Decode the generated output
            decoded_output = self.tokenizer.decode(out[0])
            answer = extract_answer(decoded_output)

            if answer == y:
                correct_predictions += 1

            if i % 3 == 0:
                print(f"Processed {i} samples. Current accuracy: {correct_predictions / (i + 1):.2f}")

        accuracy = correct_predictions / total_predictions
        print(f"Accuracy: {accuracy:.2f}")
