from typing import List

import faiss
import numpy as np

from storage.storage import Storage

class FaissStorage(Storage):
    """
    FaissStorage is a concrete implementation of the Storage abstract base class.
    It uses the Faiss library to store and query vectors efficiently.
    """

    def __init__(
            self,
            dimension: int,
            index = None,
    ):
        """
        Initializes the FaissStorage with a specified vector dimension.

        Args:
            dimension: The dimensionality of the vectors to be stored.
        """
        self.dimension = dimension
        if index is None:
            index = faiss.IndexFlatL2(dimension)
        self.index = index

    def store(self, key: str, data):
        """
        Stores a vector associated with a given key.

        Args:
            key: A unique identifier for the vector (e.g., a string).
            data: The vector to be stored. Must be a list of floats with length equal to 'dimension'.
        """
        if len(data[0]) != self.dimension:
            raise ValueError(f"Data must have {self.dimension} dimensions.")

        # Convert the data to a numpy array and ensure it's in the correct format
        data = np.array(data, dtype='float32')

        self.index.add(data)  # Add the vector to the index

    def query(self, key: str, k: int) -> (List, List):
        """
        Retrieves the vector associated with a given key.

        Args:
            key: The identifier of the vector to retrieve.

        Returns:
            A tuple containing the distances and indices of the nearest vectors.
        """
        # In this simple implementation, we don't actually use the key for retrieval.
        # In a real-world scenario, you would need to maintain a mapping from keys to indices.

        # For demonstration purposes, we'll return the first vector in the index
        if self.index.ntotal == 0:
            return None

        distances, indices = self.index.search(np.array([[0] * self.dimension], dtype='float32'), k)

        return distances[0].tolist(), indices[0].tolist()

    def export(self, file_path: str):
        """
        Exports the Faiss index to a file.

        Args:
            file_path: The path where the index will be saved.
        """
        faiss.write_index(self.index, file_path)

    def load(self, file_path: str):
        """
        Loads a Faiss index from a file.

        Args:
            file_path: The path from where the index will be loaded.
        """
        self.index = faiss.read_index(file_path)