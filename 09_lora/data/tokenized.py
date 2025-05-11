from torch.utils.data import Dataset
from data.prompt_dataset import PromptedEhvoy

class TokenizedDataset(Dataset):
    """
    Tokenized dataset for text generation tasks.
    """
    def __init__(self, data: PromptedEhvoy, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        y = f"{x} {y}"

        x_tokenized = self.tokenizer(
            x,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        y_tokenized = self.tokenizer(
            y,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return x_tokenized, y_tokenized

