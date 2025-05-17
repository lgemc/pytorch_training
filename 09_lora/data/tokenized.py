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
        x, _ = self.data[idx]

        x_tokenized = self.tokenizer(
            x,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        x_tokenized["labels"] = x_tokenized["input_ids"].copy()

        return x_tokenized

