from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    def __init__(self, tokenizer, dataset: Dataset, max_length: int = 800):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        tokenized_item = self.tokenizer(
            item["content"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized_item["id"] = item["id"]
        tokenized_item["PMID"] = item["PMID"]
        return tokenized_item