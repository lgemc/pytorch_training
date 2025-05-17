import unittest
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM
import torch

from data.ehovy_race import  EhovyRaceDataset
from data.prompt_dataset import PromptedEhvoy
from data.tokenized import TokenizedDataset

from model.llama_3_2_tokenizer import Llama32Tokenizer

from train.lora import train

device = "cuda" if torch.cuda.is_available() else "cpu"
def get_dataset(tokenizer, split):
    """
    Get the training dataset.
    """
    ehovy_dataset = EhovyRaceDataset(variation="high", split=split, max_article_size=800)
    prompted_dataset = PromptedEhvoy(ehovy_dataset, include_answer=True)
    tokenized_dataset = TokenizedDataset(prompted_dataset, tokenizer)
    return tokenized_dataset

class TestLora(unittest.TestCase):
    def test_lora(self):
        # Initialize the tokenizer and model
        model_1B = "meta-llama/Llama-3.2-1B"
        tokenizer = Llama32Tokenizer(model_1B)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer.tokenizer, mlm=False)
        model = AutoModelForCausalLM.from_pretrained(model_1B, torch_dtype=torch.float16).to(device)

        # Initialize the dataset
        train_dataset = get_dataset(tokenizer, split="train")
        val_dataset = get_dataset(tokenizer, split="validation")
        # Train the LoRA model
        train(
            model, data_collator, train_dataset, val_dataset,
            num_epochs=1, batch_size=8, learning_rate=5e-5,
            device=device
        )