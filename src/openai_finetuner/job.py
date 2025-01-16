"""Manages OpenAI fine-tuning jobs with caching based on hyperparameters."""

import json
import os
import pathlib
import hashlib
from typing import Optional, Dict, Any

from openai import OpenAI

from .constants import get_cache_dir
from .interfaces import JobManagerInterface, JobInfo

class JobManager(JobManagerInterface):
    def __init__(self, api_key: Optional[str] = None, base_dir: pathlib.Path = get_cache_dir() / ".finetune_jobs"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            
        self.base_dir = pathlib.Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file = self.base_dir / "jobs.json"
        self.client = OpenAI(api_key=self.api_key)
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

    def create_job(self, 
                  training_file: str,
                  model: str,
                  validation_file: Optional[str] = None,
                  hyperparameters: Optional[Dict[str, Any]] = None,
                  suffix: Optional[str] = None) -> JobInfo:
        """
        Create a new fine-tuning job or return existing one with same config.
        
        Args:
            training_file: File ID for training data
            model: Base model to fine-tune
            validation_file: Optional file ID for validation data
            hyperparameters: Optional hyperparameters dict
            suffix: Optional suffix for the fine-tuned model name
            
        Returns:
            JobInfo object containing job details
        """
        # Compute hash of job config
        config_hash = self._compute_hash(
            training_file=training_file,
            model=model,
            validation_file=validation_file,
            hyperparameters=hyperparameters,
            suffix=suffix
        )
        
        # Check if job with identical config exists
        if config_hash in self.jobs:
            job_id = self.jobs[config_hash]
            # Get all jobs and find matching one
            jobs = self.client.fine_tuning.jobs.list()
            for job in jobs.data:
                if job.id == job_id:
                    return JobInfo(
                        id=job.id,
                        model=job.model,
                        training_file=job.training_file,
                        hyperparameters=job.hyperparameters,
                        status=job.status,
                        fine_tuned_model=job.fine_tuned_model
                    )
            
            # Job not found in list, remove from cache
            del self.jobs[config_hash]
            self._save_jobs()
            
        # Create new job
        create_args = {
            "training_file": training_file,
            "model": model
        }
        if validation_file:
            create_args["validation_file"] = validation_file
        if hyperparameters:
            create_args["hyperparameters"] = hyperparameters
        if suffix:
            create_args["suffix"] = suffix
            
        response = self.client.fine_tuning.jobs.create(**create_args)
        
        # Save job ID with config hash
        self.jobs[config_hash] = response.id
        self._save_jobs()
        
        return JobInfo(
            id=response.id,
            model=response.model,
            training_file=response.training_file,
            hyperparameters=response.hyperparameters,
            status=response.status,
            fine_tuned_model=response.fine_tuned_model
        )
