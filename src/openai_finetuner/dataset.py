"""Registry for managing datasets and their associated IDs and hashes."""

import hashlib
import json
import os
import pathlib
from typing import Optional, List, Dict
from .openai.types import Example
from .constants import get_cache_dir

DatasetID = str
DatasetHash = str

class DatasetRegistry:
    def __init__(self, base_dir: str = ".datasets") -> None:
        self.base_dir: pathlib.Path = get_cache_dir() / base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.id_map_file: pathlib.Path = self.base_dir / "id_map.json"
        self.id_to_hash: Dict[DatasetID, DatasetHash] = {}
        self._load_id_map()

    def _load_id_map(self) -> None:
        """Load the ID to hash mapping from disk."""
        if self.id_map_file.exists():
            with open(self.id_map_file) as f:
                self.id_to_hash = json.load(f)
        else:
            self.id_to_hash = {}

    def _save_id_map(self) -> None:
        """Save the ID to hash mapping to disk."""
        # Write to temporary file first to ensure atomic operation
        temp_file = self.id_map_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.id_to_hash, f, indent=2)
        os.replace(temp_file, self.id_map_file)

    def compute_hash(self, dataset: List[Example]) -> DatasetHash:
        """Compute a stable hash for a dataset based on its content."""
        # Sort the dataset to ensure stable hashing
        serialized = json.dumps(dataset, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _get_path_for_hash(self, hash_value: DatasetHash) -> pathlib.Path:
        """Get the file path for a dataset with the given hash."""
        return self.base_dir / f"{hash_value}.json"

    def save_dataset(self, dataset: List[Example], dataset_id: Optional[DatasetID] = None) -> DatasetHash:
        """
        Save a dataset and optionally associate it with an ID.
        Returns the hash of the dataset.
        """
        hash_value = self.compute_hash(dataset)
        dataset_path = self._get_path_for_hash(hash_value)

        if not dataset_path.exists():
            # Write to temporary file first to ensure atomic operation
            temp_path = dataset_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            os.replace(temp_path, dataset_path)

        if dataset_id is not None:
            self.id_to_hash[dataset_id] = hash_value
            self._save_id_map()

        return hash_value

    def load_dataset(self, identifier: str) -> List[Example]:
        """
        Load a dataset by its ID or hash.
        Raises FileNotFoundError if dataset doesn't exist.
        """
        hash_value = self.id_to_hash.get(identifier, identifier)
        dataset_path = self._get_path_for_hash(hash_value)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {identifier}")
            
        with open(dataset_path) as f:
            return json.load(f)

    def get_hash(self, dataset_id: DatasetID) -> Optional[DatasetHash]:
        """Get the hash associated with a dataset ID."""
        return self.id_to_hash.get(dataset_id)

    def associate_id(self, dataset_id: DatasetID, hash_value: DatasetHash) -> None:
        """Associate an ID with an existing dataset hash."""
        if not self._get_path_for_hash(hash_value).exists():
            raise FileNotFoundError(f"No dataset exists with hash: {hash_value}")
        
        self.id_to_hash[dataset_id] = hash_value
        self._save_id_map()
