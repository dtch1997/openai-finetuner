"""Manages datasets for fine-tuning, including validation and formatting."""

import json
import pathlib
from typing import Any, Union

from .constants import get_cache_dir

Message = dict[str, Any]
Dataset = list[Message]

# TODO: Improve DatasetManager to also store a hash of the file content
# And if the file content changes, update the dataset
class DatasetManager:
    """Manages the storage, retrieval, and manipulation of fine-tuning datasets.

    This class provides functionality to create, store, retrieve, and manage datasets
    used for fine-tuning language models. It handles dataset storage in JSONL format
    and provides a consistent interface for dataset operations.

    Attributes:
        base_dir (pathlib.Path): Base directory for all cache data
        datasets_dir (pathlib.Path): Directory specifically for dataset storage

    Example:
        >>> manager = DatasetManager()
        >>> dataset = [{"messages": [...]}]
        >>> path = manager.create_dataset("my_dataset", dataset)
        >>> retrieved = manager.retrieve_dataset("my_dataset")
        >>> manager.list_datasets()
        ['my_dataset']
    """

    def __init__(
        self,
        base_dir: pathlib.Path | None = None
    ):
        self.base_dir = base_dir or get_cache_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir = self.base_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)

    def create_dataset(
        self,
        id: str,
        dataset_or_file: Union[str, bytes, pathlib.Path, Dataset]
    ) -> pathlib.Path:
        """
        Create a dataset from input file or dict and save it.

        Args:
            id: Unique identifier for the dataset
            file: Dataset file path, bytes, or dict containing messages

        Returns:
            Path to the saved dataset file
        """
        dataset_path = self.get_dataset_path(id)

        # Check if dataset already exists
        if dataset_path.exists():
            return dataset_path

        # Handle different input types
        if isinstance(dataset_or_file, (str, pathlib.Path)):
            # Copy existing file
            with open(dataset_or_file) as f:
                data = json.load(f)
        elif isinstance(dataset_or_file, bytes):
            data = json.loads(dataset_or_file)
        elif isinstance(dataset_or_file, list):
            data = dataset_or_file
        else:
            raise ValueError("Dataset must be path, bytes or list of messages")

        # Validate and write messages to JSONL
        with open(dataset_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        return dataset_path

    def get_dataset_path(self, id: str) -> pathlib.Path:
        return self.datasets_dir / f"{id}.jsonl"

    def retrieve_dataset(self, id: str) -> Dataset:
        dataset_path = self.get_dataset_path(id)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {id} not found")
        with open(dataset_path) as f:
            return [json.loads(line) for line in f]

    def remove_dataset(self, id: str) -> None:
        dataset_path = self.get_dataset_path(id)
        if dataset_path.exists():
            dataset_path.unlink()

    def list_datasets(self) -> list[str]:
        return [f.stem for f in self.datasets_dir.glob("*.jsonl")]
