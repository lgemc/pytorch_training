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
            nlist: int = 100,
            data_to_use_for_training: int = 1000,
    ):
        """
        Initializes the FaissStorage with a specified vector dimension.

        Args:
            dimension: The dimensionality of the vectors to be stored.
        """
        self.dimension = dimension
        quantizer = faiss.IndexFlatL2(dimension)  # the standard quantizer
        self._train_buffer = []
        self.data_to_use_for_training = data_to_use_for_training

        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

    def store(self, key: str, data):
        """
        Stores a vector associated with a given key.

        Args:
            key: A unique identifier for the vector (e.g., a string).
            data: The vector to be stored. Must be a list of floats with length equal to 'dimension'.
        """
        if len(data) != self.dimension:
            raise ValueError(f"Data must have {self.dimension} dimensions.")

        # Convert the data to a numpy array and ensure it's in the correct format
        data = np.array(data, dtype='float32')
        if not self.index.is_trained:
            if len(self._train_buffer) < self.data_to_use_for_training:
                self._train_buffer.append(data)

                return

            print("Training the index with the training data...")
            # Train the index with the training data
            buffer = np.array(self._train_buffer, dtype='float32')
            print(buffer.shape)
            self.index.train(buffer)
            self.index.add(buffer)  # Add the vector to the index
            self._train_buffer = []
            print("Index trained successfully.")
            return

        data = np.array([data])  # Wrap the data in a list to match the expected input shape
        self.index.add(data)  # Add the vector to the index

    def query(self, key: str) -> list:
        """
        Retrieves the vector associated with a given key.

        Args:
            key: The identifier of the vector to retrieve.

        Returns:
            The stored vector as a list of floats.
        """
        # In this simple implementation, we don't actually use the key for retrieval.
        # In a real-world scenario, you would need to maintain a mapping from keys to indices.

        # For demonstration purposes, we'll return the first vector in the index
        if self.index.ntotal == 0:
            return None

        distances, indices = self.index.search(np.array([[0] * self.dimension], dtype='float32'), 1)

        return self.index.reconstruct(indices[0][0]).tolist()  # Return as list of floats

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