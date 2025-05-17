from typing import Callable, Tuple, List

from torch.utils.data import Dataset

from storage.storage import Storage

querier_type = Callable[[str, int], Tuple[List, List]]

def build_querier(
        storage: Storage,
        original_dataset: Dataset,
        tokenizer_fn: callable,
) -> querier_type:
    """
    Build a query function that can be used to query the storage system.

    Args:
        storage (Storage): The storage system to use for querying.
        original_dataset (Dataset): The original dataset to use for querying.
        tokenizer_fn (callable): The function to use for tokenizing the query.

    Returns:
        callable: A function that takes a query string and returns the results.
    """
    def querier(query: str, k: int = 10) -> (List, List):
        return perform_query(storage, original_dataset, tokenizer_fn, query, k)

    return querier

def perform_query(
        storage: Storage,
        original_dataset: Dataset,
        tokenizer_fn: callable,
        query: str,
        k: int,
) -> (List, List):
    query_vector = tokenizer_fn(query)
    distances, indices = storage.query(query_vector, k)

    data = []
    for index in indices:
        data.append(original_dataset[index])

    return distances, data

