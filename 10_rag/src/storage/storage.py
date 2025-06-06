from typing import List

from abc import ABC, abstractmethod
from typing import Any, Optional

class Storage(ABC):
    """
    Abstract base class defining a template for storage systems.
    Subclasses must implement the 'store' and 'query' methods.
    """

    @abstractmethod
    def store(self, key: str, data: Any):
        """
        Stores data associated with a given key.

        Args:
            key: A unique identifier for the data (e.g., a string, int).
            data: The data to be stored. Can be any Python object
                  that can be serialized or handled by the specific storage.
        """
        # Abstract methods typically don't have an implementation in the base class
        # You can use 'pass' or raise a NotImplementedError, though 'pass' is common with @abstractmethod
        pass

    @abstractmethod
    def query(self, key: str, k: int) -> (List, List):
        """
        Retrieves data associated with a given key.

        Args:
            key: The identifier of the data to retrieve.
            k: The number of nearest vectors to retrieve.

        Returns:
            A tuple containing the distances and indices of the nearest vectors.
        """
        pass