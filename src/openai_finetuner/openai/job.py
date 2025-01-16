"""Manages OpenAI fine-tuning jobs with caching based on hyperparameters."""

import json
import os
import pathlib
import hashlib
from typing import Optional, Dict, Any

from openai.types.fine_tuning.fine_tuning_job import FineTuningJob
from openai.types.fine_tuning.jobs.fine_tuning_job_checkpoint import FineTuningJobCheckpoint

from ..constants import get_cache_dir
from ..core.interfaces import JobManagerInterface, CheckpointManagerInterface
from .client import ClientManager

client_manager = ClientManager()

class JobManager(JobManagerInterface, CheckpointManagerInterface):
    def __init__(
        self,
        base_dir: pathlib.Path = get_cache_dir()
    ):
        self.base_dir = pathlib.Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file = self.base_dir / "jobs.json"
        self._load_jobs()

    def _load_jobs(self):
        """Load the jobs registry from disk."""
        if self.jobs_file.exists():
            with open(self.jobs_file) as f:
                self.jobs = json.load(f)
        else:
            self.jobs = {}

    def _save_jobs(self):
        """Save the jobs registry to disk."""
        temp_file = self.jobs_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.jobs, f, indent=2)
        os.replace(temp_file, self.jobs_file)

    def _compute_hash(self, **kwargs) -> str:
        """Compute a stable hash of the job arguments."""
        # Sort to ensure stable hashing
        config_str = json.dumps(kwargs, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def create_job(
        self, 
        file_id: str,
        model: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None
    ) -> FineTuningJob:
        """
        Create a new fine-tuning job or return existing one with same config.
        
        Args:
            training_file: File ID for training data
            model: Base model to fine-tune
            validation_file: Optional file ID for validation data
            hyperparameters: Optional hyperparameters dict
            suffix: Optional suffix for the fine-tuned model name
            
        Returns:
            FineTuningJob object
        """
        # Compute hash of job config
        config_hash = self._compute_hash(
            file_id=file_id,
            model=model,
            hyperparameters=hyperparameters,
            suffix=suffix
        )
        
        # Check if job with identical config exists
        if config_hash in self.jobs:
            job_id = self.jobs[config_hash]
            # Get all jobs and find matching one
            jobs = client_manager.client.fine_tuning.jobs.list()
            if job_id in [job.id for job in jobs]:
                return client_manager.client.fine_tuning.jobs.retrieve(job_id)
            
            # Job not found in list, remove from cache
            del self.jobs[config_hash]
            self._save_jobs()
            
        # Create new job
        create_args = {
            "training_file": file_id,
            "model": model
        }
        if hyperparameters:
            create_args["hyperparameters"] = hyperparameters
        if suffix:
            create_args["suffix"] = suffix
            
        response = client_manager.client.fine_tuning.jobs.create(**create_args)
        
        # Save job ID with config hash
        self.jobs[config_hash] = response.id
        self._save_jobs()
        
        return response

    def get_job(self, job_id: str) -> FineTuningJob:
        return client_manager.client.fine_tuning.jobs.retrieve(job_id)

    def get_checkpoint(self, job_id: str) -> FineTuningJobCheckpoint | None:
        checkpoints = self.list_checkpoints(job_id)
        if not checkpoints:
            return None
        # Get the checkpoint with the highest step number 
        last_checkpoint = max(checkpoints, key=lambda x: x.step_number)
        return last_checkpoint
    
    def list_checkpoints(self, job_id: str) -> list[FineTuningJobCheckpoint]:
        checkpoints = client_manager.client.fine_tuning.jobs.checkpoints.list(job_id)
        return checkpoints.data
