"""Manages file uploads to OpenAI API with local caching."""

import pathlib
import hashlib
import json
import os

from openai.types import FileObject

from ..core.interfaces import FileManagerInterface
from ..constants import get_cache_dir
from .client import ClientManager

client_manager = ClientManager()

class FileManager(FileManagerInterface):
    def __init__(
        self,
        base_dir: pathlib.Path = get_cache_dir()
    ):
        self.base_dir = pathlib.Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.base_dir / "files.json"
        self._load_cache()

    def _load_cache(self):
        """Load the file cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def _save_cache(self):
        """Save the file cache to disk."""
        temp_file = self.cache_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
        os.replace(temp_file, self.cache_file)

    def _calculate_hash(self, file: str | bytes | pathlib.Path) -> str:
        """Calculate a deterministic hash of the file contents."""
        file_contents = file.read_bytes() if isinstance(file, pathlib.Path) else file
        return hashlib.sha256(file_contents).hexdigest()

    def create_file(
        self,
        file: str | bytes | pathlib.Path,
        purpose: str = "fine-tune"
    ) -> FileObject:
        """
        Upload a file to OpenAI API, using caching if it was previously uploaded.
        Returns FileObject.
        """
        file_hash = self._calculate_hash(file)

        # Check if file exists in OpenAI
        if file_hash in self.cache:
            file_id = self.cache[file_hash]
            # Verify file still exists in OpenAI by getting list of files
            files = client_manager.client.files.list()
            if file_id in [file.id for file in files]:
                return client_manager.client.files.retrieve(file_id)

            else:
                # File no longer exists, remove from cache
                del self.cache[file_hash]
                self._save_cache()

        # Upload new file
        response = client_manager.client.files.create(purpose=purpose, file=file)
        
        # Cache the file ID
        self.cache[file_hash] = response.id
        self._save_cache()

        return response

    def get_file(self, file_id: str) -> FileObject:
        return client_manager.client.files.retrieve(file_id)
