import json

from torch.utils.data import Dataset

class FromJsonDataset(Dataset):
    def __init__(self, json_file):
        self.raw_content = ""
        with open(json_file, "r") as f:
            self.raw_content = f.read()

        self.data = json.loads(self.raw_content)

    def __len__(self):
        return len(self.data["id"])

    def __getitem__(self, idx: int):
        return  {
            "title": self.data["title"][idx],
            "content": self.data["content"][idx],
            "contents": self.data["contents"][idx],
            "PMID": self.data["PMID"][idx],
            "id": self.data["id"][idx],
        }