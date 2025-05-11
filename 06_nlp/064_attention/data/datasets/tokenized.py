import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import re
import random

class TokenizedDataset(Dataset):
    def __init__(
            self,
            file_name: str,
            tokenizer=None,
            batch_size_words=100,
            max_token_length=600,
            amount_of_samples=None,
    ):
        self.batch_size_words = batch_size_words
        self.max_token_length = max_token_length
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.tokenizer.pad_token = self.tokenizer.eos_token

        with open(file_name) as file:
            self._raw_content = file.read()

        # Split text into words while preserving whitespace
        self.words_with_spaces = re.findall(r'\S+|\s+', self._raw_content)
        # Create word batches and tokenize them
        self.tokenized_batches = []
        self.batch_start_indices = []

        # Create a single tokenized tensor for the entire text
        self.tokenized = self.tokenizer.encode(self._raw_content)
        self.tokenized = torch.tensor(self.tokenized).long()

        self._vocab_size = len(self.tokenizer)

        self._indexes = []
        for i in range(0, len(self.tokenized) - self.max_token_length, self.max_token_length):
            self._indexes.append(i)

        # If there's a remainder, include it as the last sequence
        if len(self.tokenized) - self.max_token_length > 0:
            last_valid_start = len(self.tokenized) - self.max_token_length
            if last_valid_start not in self._indexes:
                self._indexes.append(last_valid_start)

        if amount_of_samples is not None:
            if amount_of_samples < len(self._indexes):
                pass
            else:
                for i in range(len(self._indexes), amount_of_samples):
                    self._indexes.append(random.randint(0, len(self.tokenized) - self.max_token_length))

        # Shuffle the indexes
        random.shuffle(self._indexes)

        self._num_sequences = len(self._indexes)

    def __len__(self):
        return self._num_sequences

    def __getitem__(self, idx):
        start_idx = self._indexes[idx]
        end_idx = min(start_idx + self.max_token_length, len(self.tokenized))

        # Get input sequence
        x = self.tokenized[start_idx:end_idx]

        # If the sequence is shorter than max_token_length, pad it
        if len(x) < self.max_token_length:
            padding = torch.full((self.max_token_length - len(x),), self.tokenizer.pad_token_id)
            x = torch.cat([x, padding])

        # Get target sequence (shifted by 1)
        y_start = start_idx + 1
        y_end = min(y_start + self.max_token_length, len(self.tokenized))
        y = self.tokenized[y_start:y_end]

        # Pad y if necessary
        if len(y) < self.max_token_length:
            print(f"padded tokens")
            padding = torch.full((self.max_token_length - len(y),), self.tokenizer.pad_token_id)
            y = torch.cat([y, padding])

        return x, y

    @property
    def vocab_size(self):
        return self._vocab_size