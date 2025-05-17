import sqlite3
from typing import List, Tuple, Any
from storage.storage import Storage
import numpy as np

import sqlite_vss

class SqliteStorage(Storage):
    """
    SqliteStorage is a concrete implementation of the Storage abstract base class.
    It uses SQLite with the sqlite-vss extension to store and query vector embeddings.
    """

    def __init__(
            self,
            db_path: str,
            dimension: int,
            table_name: str = "document_vectors"
    ):
        """
        Initializes the SqliteStorage. Connects to the database,
        loads the sqlite-vss extension, and creates the vector table if it doesn't exist.

        Args:
            db_path: Path to the SQLite database file.
            dimension: The dimensionality of the vectors to be stored.
            table_name: The name for the virtual FTS5 table.
        """
        self.db_path = db_path
        self.dimension = dimension
        self.table_name = table_name
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_table_if_not_exists()

    def _connect(self):
        """Establishes database connection and loads the vss extension."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.enable_load_extension(True)
            sqlite_vss.load(self.conn)
            self.cursor = self.conn.cursor()

            # Attempt to load the sqlite-vss extension
            # The exact extension name or path might vary depending on your installation
            self.conn.enable_load_extension(True)
            try:
                # Try common extension names
                self.conn.load_extension("vss0")
                # print("sqlite-vss extension 'vss0' loaded.")
            except sqlite3.OperationalError:
                 try:
                    # Try another common name/path might be needed
                    # You might need to specify the full path to the shared library (.so, .dll, .dylib)
                    # self.conn.load_extension("/path/to/your/vss0.so") # Example with explicit path
                    self.conn.load_extension("vss") # Sometimes just 'vss' works
                    # print("sqlite-vss extension 'vss' loaded.")
                 except sqlite3.OperationalError as e:
                    print(f"Warning: Could not load sqlite-vss extension. Vector search will not work. Error: {e}")


        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            self.conn = None
            self.cursor = None
            raise

    def _create_table_if_not_exists(self):
        """Creates the sqlite-vss virtual table if it doesn't exist."""
        if self.conn is None:
            print("Cannot create table: Database connection not established.")
            return

        # Check if the table already exists
        self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='virtual table' AND name='{self.table_name}';")
        if self.cursor.fetchone() is None:
            try:
                # Create the virtual table using vss0
                # Stores the embedding (required for vss0) and potentially other data like original text
                self.cursor.execute(f'''
                    CREATE VIRTUAL TABLE {self.table_name} USING vss0(
                        embedding({self.dimension}),
                        original_text TEXT
                    );
                ''')
                self.conn.commit()
                print(f"sqlite-vss virtual table '{self.table_name}' created.")
            except sqlite3.OperationalError as e:
                 print(f"Error creating sqlite-vss table {self.table_name}. Is the vss0 extension loaded? {e}")
            except sqlite3.Error as e:
                print(f"Database error during table creation: {e}")
        else:
            # print(f"Table '{self.table_name}' already exists.")
            pass # Table already exists, no need to create

    def store(self, items: List[Tuple[Any, np.ndarray, str]]):
        """
        Stores a list of items, each containing a key, vector, and original text.

        Args:
            items: A list of tuples, where each tuple is (key, vector, original_text).
                   key can be any identifier, vector is a numpy array (float32),
                   and original_text is the source text for RAG.
        """
        if self.conn is None:
            print("Cannot store data: Database connection not established.")
            return

        data_to_insert = []
        for key, vector, original_text in items:
            if vector.shape[0] != self.dimension or vector.dtype != np.float32:
                 raise ValueError(f"Vector for key '{key}' must have shape ({self.dimension},) and dtype float32.")
            data_to_insert.append((vector.tobytes(), original_text)) # Prepare data for insert

        try:
            # Insert into the virtual table
            # The first column (embedding) is required for vss0
            self.cursor.executemany(f'''
                INSERT INTO {self.table_name} (embedding, original_text)
                VALUES (?, ?);
            ''', data_to_insert)
            self.conn.commit()
            # print(f"Stored {len(data_to_insert)} items into {self.table_name}.")
        except sqlite3.Error as e:
            print(f"Database error during storage: {e}")
            self.conn.rollback()


    def query(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Queries for the k most similar vectors to the query vector.

        Args:
            query_vector: The vector to query with (numpy array, float32).
            k: The number of nearest neighbors to retrieve.

        Returns:
            A list of tuples, where each tuple contains the original_text
            and the distance of a similar document. Returns empty list on error or no results.
        """
        if self.conn is None:
            print("Cannot query: Database connection not established.")
            return []

        if query_vector.shape[0] != self.dimension or query_vector.dtype != np.float32:
            raise ValueError(f"Query vector must have shape ({self.dimension},) and dtype float32.")

        query_vector_bytes = query_vector.tobytes()
        results = []

        try:
            # Use vss_search function for similarity search
            # It implicitly provides 'distance' as a column
            self.cursor.execute(f'''
                SELECT original_text, distance
                FROM {self.table_name}
                WHERE embedding MATCH vss_search(?)
                LIMIT ?;
            ''', (query_vector_bytes, k))

            results = self.cursor.fetchall()
            # print(f"Query returned {len(results)} results.")

        except sqlite3.OperationalError as e:
            print(f"Error during vector similarity search (Is sqlite-vss loaded and table correct?): {e}")
        except sqlite3.Error as e:
             print(f"Database error during query: {e}")


        return results # Returns list of (original_text, distance) tuples

    def export(self, file_path: str):
        """
        Exports the entire database to a file.

        Args:
            file_path: The path where the database file will be copied.
        """
        if self.conn is None:
            print("Cannot export: Database connection not established.")
            return

        try:
            # Close connection before copying if needed, or use a separate connection
            # For simplicity, let's just try to copy the file if connection is read-only
            # A safer way is to use a separate backup connection or handle locking.
            # However, a simple file copy while the DB is in WAL mode is often safe.
            import shutil
            shutil.copy2(self.db_path, file_path)
            # print(f"Database exported to {file_path}")
        except FileNotFoundError:
            print(f"Error exporting: Database file not found at {self.db_path}")
        except Exception as e:
            print(f"Error exporting database: {e}")


    def load(self, file_path: str):
        """
        Loads a database from a file, replacing the current one.

        Args:
            file_path: The path from where the database file will be copied.
        """
        if self.conn:
            self.conn.close() # Close existing connection

        try:
            import shutil
            shutil.copy2(file_path, self.db_path)
            # print(f"Database loaded from {file_path}")
            self._connect() # Reconnect after loading
        except FileNotFoundError:
            print(f"Error loading: Source database file not found at {file_path}")
        except Exception as e:
            print(f"Error loading database: {e}")


    def __del__(self):
        """Ensures the database connection is closed when the object is garbage collected."""
        if self.conn:
            self.conn.close()
            # print("Database connection closed automatically.")
