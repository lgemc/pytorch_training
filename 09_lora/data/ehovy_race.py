from torch.utils.data import Dataset
from datasets import load_dataset

class EhovyRaceDataset(Dataset):
    """
    Ehvoy race is a questions and answer dataset
    Variations can get the values all, high, medium and low and depending on this the dataset size may vary
    """
    def __init__(self, variation="high", split="train", max_article_size=None):
        self.raw_dataset = load_dataset("ehovy/race", variation, split=split)
        if max_article_size is not None:
            self.raw_dataset  = self.raw_dataset.filter(
                lambda example: len(example['article']) < max_article_size,
                desc=f"Filtrando artículos en {split}"
            )


    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        return self.raw_dataset[idx]