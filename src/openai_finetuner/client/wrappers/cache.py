"""Cache wrapper for API clients."""

import pathlib
import json
import os
from typing import TypeVar, Generic, Mapping, Optional, Dict, Any
import logging

from ...core.interfaces import ClientInterface, Purpose
from ...util.hash import compute_file_hash, compute_config_hash
from ...constants import get_cache_dir
from openai import RateLimitError as OpenAIRateLimitError
from ...core.errors import RateLimitError

logger = logging.getLogger(__name__)

# Type definitions
FileHash = str
JobHash = str
FileID = str
JobID = str

T = TypeVar('T', bound=ClientInterface)

class CacheWrapper(ClientInterface, Generic[T]):
    """Wrapper that adds caching to any ClientInterface implementation."""
    
    files_cache: Mapping[FileHash, FileID]
    jobs_cache: Mapping[JobHash, JobID]
    files_cache_path: pathlib.Path
    jobs_cache_path: pathlib.Path
    base_dir: pathlib.Path
    client: T

    def __init__(
        self,
        client: T,
        base_dir: pathlib.Path = get_cache_dir()
    ):
        self.client = client
        self.base_dir = pathlib.Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.files_cache_path = self.base_dir / "files.json"
        self.jobs_cache_path = self.base_dir / "jobs.json"
        
        # Load caches
        self._load_caches()

    def _load_caches(self):
        """Load both file and job caches from disk."""
        # Load files cache
        if self.files_cache_path.exists():
            with open(self.files_cache_path) as f:
                self.files_cache = json.load(f)
        else:
            self.files_cache = {}

        # Load jobs cache
        if self.jobs_cache_path.exists():
            with open(self.jobs_cache_path) as f:
                self.jobs_cache = json.load(f)
        else:
            self.jobs_cache = {}

    def _save_cache(self, cache_path: pathlib.Path, cache_data: dict):
        """Save a cache to disk safely using atomic write."""
        temp_file = cache_path.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        os.replace(temp_file, cache_path)

    def create_file(self, file: str | bytes | pathlib.Path, purpose: Purpose = "fine-tune"):
        """Create a file with caching."""
        file_hash: FileHash = compute_file_hash(file)

        # Check cache for existing file
        if file_hash in self.files_cache:
            file_id = self.files_cache[file_hash]
            try:
                return self.client.get_file(file_id)
            except Exception as _:
                # File no longer exists, remove from cache
                del self.files_cache[file_hash]
                self._save_cache(self.files_cache_path, self.files_cache)

        # Upload new file
        try:
            response = self.client.create_file(file, purpose)
        except OpenAIRateLimitError as e:
            raise RateLimitError(e)
        
        # Update cache
        self.files_cache[file_hash] = response.id
        self._save_cache(self.files_cache_path, self.files_cache)

        return response

    def get_file(self, file_id: str):
        """Pass through to client."""
        try: 
            return self.client.get_file(file_id)
        except Exception as e:
            # File not found, remove from cache
            del self.files_cache[file_id]
            self._save_cache(self.files_cache_path, self.files_cache)
            raise e

    def create_job(
        self,
        file_id: str,
        model: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None
    ):
        """Create a job with caching."""
        job_hash: JobHash = compute_config_hash(
            file_id=file_id,
            model=model,
            hyperparameters=hyperparameters,
            suffix=suffix
        )

        # Check cache for existing job
        if job_hash in self.jobs_cache:
            job_id = self.jobs_cache[job_hash]
            try:
                return self.client.get_job(job_id)
            except Exception as e:
                # Job not found, remove from cache
                del self.jobs_cache[job_hash]
                self._save_cache(self.jobs_cache_path, self.jobs_cache)
                raise e

        # Create new job
        response = self.client.create_job(file_id, model, hyperparameters, suffix)
        
        # Update cache
        self.jobs_cache[job_hash] = response.id
        self._save_cache(self.jobs_cache_path, self.jobs_cache)

        return response

    def get_job(self, job_id: str):
        """Pass through to client."""
        try:
            return self.client.get_job(job_id)
        except Exception as e:
            # Job not found, remove from cache
            del self.jobs_cache[job_id]
            self._save_cache(self.jobs_cache_path, self.jobs_cache)
            raise e

    def get_checkpoint(self, job_id: str):
        """Pass through to client."""
        return self.client.get_checkpoint(job_id)

    def list_checkpoints(self, job_id: str):
        """Pass through to client."""
        return self.client.list_checkpoints(job_id) 