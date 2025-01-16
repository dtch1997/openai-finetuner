"""Manages file uploads to OpenAI API with local caching."""

import json
import os
import pathlib
from typing import Optional

from openai import OpenAI

from .dataset import DatasetRegistry
from .interfaces import FileManagerInterface, FileInfo
from .constants import get_cache_dir

class FileManager(FileManagerInterface):
    def __init__(self, api_key: Optional[str] = None, base_dir: pathlib.Path = get_cache_dir() / ".openai_files"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            
        self.base_dir = pathlib.Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.file_map_path = self.base_dir / "file_map.json"
        self.dataset_registry = DatasetRegistry()
        self.client = OpenAI(api_key=self.api_key)
        self._load_file_map()

    def _load_file_map(self):
        """Load the hash to file ID mapping from disk."""
        if self.file_map_path.exists():
            with open(self.file_map_path) as f:
                self.hash_to_file = json.load(f)
        else:
            self.hash_to_file = {}

    def _save_file_map(self):
        """Save the hash to file ID mapping to disk."""
        temp_file = self.file_map_path.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.hash_to_file, f, indent=2)
        os.replace(temp_file, self.file_map_path)

    def upload_dataset(self, dataset_id: str, purpose: str = "fine-tune") -> FileInfo:
        """
        Upload a dataset to OpenAI API, using caching if it was previously uploaded.
        Returns FileInfo containing the file ID and dataset hash.
        """
        # Get dataset hash
        dataset_hash = self.dataset_registry.get_hash(dataset_id)
        if not dataset_hash:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Check if already uploaded
        if dataset_hash in self.hash_to_file:
            return FileInfo(
                file_id=self.hash_to_file[dataset_hash],
                dataset_hash=dataset_hash
            )

        # Load dataset and create temp JSONL file
        dataset = self.dataset_registry.load_dataset(dataset_id)
        temp_file = self.base_dir / f"{dataset_hash}.jsonl"
        
        with open(temp_file, 'w') as f:
            for example in dataset:
                f.write(json.dumps(example) + '\n')

        try:
            # Upload to OpenAI using the Files API
            with open(temp_file, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose=purpose
                )
            file_id = response.id

            # Save mapping
            self.hash_to_file[dataset_hash] = file_id
            self._save_file_map()

            return FileInfo(file_id=file_id, dataset_hash=dataset_hash)

        finally:
            # Cleanup
            temp_file.unlink()
