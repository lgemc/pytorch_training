from typing import Callable, List, Optional
import torch
from torch.utils.data import Dataset

Prompted = Callable[[str, List[str], Optional[List[str]]], str]

class Tokenized(Dataset):
    def __init__(self, tokenizer, dataset: Prompted, max_length=2000):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        text, answer = self.dataset[idx]

        result = self.tokenizer(text, padding="max_length", truncation=True, max_len=self.max_length)
        result["labels"] = torch.tensor(answer, dtype=torch.long)

        return result