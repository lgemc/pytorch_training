from torch.utils.data import Dataset

from data.pubmed.from_json import FromJsonDataset

class ContentsDataset(Dataset):
    def __init__(self, dataset: FromJsonDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        return item["contents"]