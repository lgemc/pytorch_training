import json

from torch.utils.data import Dataset

class TrainAndEval(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._raw_data = []
        with open(file_path, "r") as f:
            for line in f:
                self._raw_data.append(line.strip())

    def __len__(self):
        return len(self._raw_data)

    def __getitem__(self, idx: int):
        return json.loads(self._raw_data[idx])
