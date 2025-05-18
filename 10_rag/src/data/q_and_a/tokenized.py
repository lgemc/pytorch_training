import torch
from torch.utils.data import Dataset

from data.q_and_a.prompted import Prompted

class Tokenized(Dataset):
    def __init__(self, tokenizer, dataset: Prompted):
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        text, answer = self.dataset[idx]

        result = self.tokenizer(text, truncation=False)
        result["labels"] = torch.tensor(answer, dtype=torch.long)

        return result