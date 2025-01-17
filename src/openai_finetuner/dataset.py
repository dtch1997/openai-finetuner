"""Manages datasets for fine-tuning, including validation and formatting."""

import json
import pathlib
from typing import Any, Union

from .constants import get_cache_dir

Message = dict[str, Any]
Dataset = list[Message]

class DatasetManager:
    def __init__(self, base_dir: pathlib.Path = get_cache_dir()):
        self.base_dir = pathlib.Path(base_dir)
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
        dataset_path = self.datasets_dir / f"{id}.jsonl"

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
