from torch.utils.data import Dataset
from storage.faiss_ import FaissStorage

def index(
        data: Dataset, # The dataset to index, it should be tokenized
        storage: FaissStorage,
        data_transform: callable = None,
):
    """
    Indexes the dataset into the storage system.

    Args:
        data (Dataset): The dataset to index, it should be tokenized.
        storage (Storage): The storage system to use for indexing.
    """
    for i in range(len(data)):
        item = data[i]
        key = item["id"]
        if data_transform:
            item = data_transform(item)
        storage.store(key, item)
        if i % 1000 == 0:
            print(f"Indexed {i} items")

    return storage