from torch.utils.data import Dataset
from datasets import load_dataset

pub_med_dataset = "MedRAG/pubmed"

class PubMed(Dataset):
    def __init__(self, split):
        self.data = load_dataset(pub_med_dataset, split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

