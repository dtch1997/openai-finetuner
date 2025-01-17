"""Manages experiments that coordinate dataset, file, job and model management."""


from typing import Optional, Dict, Any
import json
import pathlib

from openai import RateLimitError
from .openai.file import FileManager
from .openai.job import JobManager
from .openai.client import ClientManager
from .core.interfaces import (
    FileManagerInterface, 
    JobManagerInterface, 
    ExperimentManagerInterface,
    CheckpointManagerInterface
)
from .core.types import ( 
    ExperimentInfo, 
    JobInfo, 
    FileInfo,
    CheckpointInfo
)
from .system.key import KeyManager
from .constants import get_cache_dir
from .dataset import DatasetManager

key_manager = KeyManager()
dataset_manager = DatasetManager()
client_manager = ClientManager()

class JobFailedError(Exception):
    """Raised when a fine-tuning job fails."""
    def __init__(self, job_id: str, status: str, error: Optional[str] = None):
        self.job_id = job_id
        self.status = status
        self.error = error
        message = f"Job {job_id} failed with status {status}"
        if error:
            message += f": {error}"
        super().__init__(message)

available_keys = key_manager.list_keys()
key_manager.set_key(available_keys.pop(0))

class ExperimentManager(ExperimentManagerInterface):
    def __init__(
        self,
        file_manager: Optional[FileManagerInterface] = None,
        job_manager: Optional[JobManagerInterface] = None,
        checkpoint_manager: Optional[CheckpointManagerInterface] = None,
        base_dir: pathlib.Path = get_cache_dir()
    ):
        self.file_manager = file_manager or FileManager()
        self.job_manager = job_manager or JobManager()
        self.checkpoint_manager = checkpoint_manager or JobManager()
        self.base_dir = pathlib.Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.base_dir / "experiments.json"
        self._load_experiments()

    def _load_experiments(self):
        """Load the experiments registry from disk."""
        if self.experiments_file.exists():
            with open(self.experiments_file) as f:
                self.experiments = json.load(f)
        else:
            self.experiments = {}

    def _save_experiments(self):
        """Save the experiments registry to disk."""
        temp_file = self.experiments_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        pathlib.Path(temp_file).replace(self.experiments_file)

    def create_experiment(
        self,
        dataset_id: str,
        base_model: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> ExperimentInfo:
        """
        Create and run a fine-tuning experiment.
        
        Args:
            name: Name of the experiment
            dataset_id: ID of dataset to use for training
            base_model: Base model to fine-tune
            hyperparameters: Optional hyperparameters for fine-tuning
            
        Returns:
            ExperimentInfo containing details about the experiment
        """
        # Check if experiment exists
        if name in self.experiments:
            return ExperimentInfo.from_dict(self.experiments[name])

        try: 
            # Upload dataset file
            file_info = self.file_manager.create_file(
                file=dataset_manager.get_dataset_path(dataset_id),
            )

            # Create fine-tuning job
            job_info = self.job_manager.create_job(
                file_id=file_info.id,
                model=base_model,
                hyperparameters=hyperparameters,
                suffix=name
            )

            if job_info.status == "failed":
                raise JobFailedError(job_info.id, job_info.status, job_info.error)

            # Create experiment info
            experiment_info = ExperimentInfo(
                name=name,
                dataset_id=dataset_id,
                base_model=base_model,
                file_id=file_info.id,
                job_id=job_info.id,
                hyperparameters=hyperparameters,
                api_key_name=key_manager.get_key()
            )

            # Save experiment
            self.experiments[name] = experiment_info.to_dict()
            self._save_experiments()

            return experiment_info
        
        except RateLimitError as e:
            # job_manager.create_job failed due to rate limit
            if len(available_keys) == 0:
                # All keys have been exhausted
                raise e
            
            # Use next available key
            key_manager.set_key(available_keys.pop(0))
            return self.create_experiment(
                dataset_id, 
                base_model, 
                hyperparameters, 
                name
            )
        
        except Exception as e:
            raise e

    def get_experiment_info(self, experiment_name: str) -> ExperimentInfo:
        return ExperimentInfo.from_dict(self.experiments[experiment_name])

    def get_job_info(self, experiment_name: str) -> JobInfo:
        experiment_info = self.get_experiment_info(experiment_name)
        return self.job_manager.get_job(experiment_info.job_id)

    def get_file_info(self, experiment_name: str) -> FileInfo:
        experiment_info = self.get_experiment_info(experiment_name)
        return self.file_manager.get_file(experiment_info.file_id)
    
    def list_experiments(self) -> list[ExperimentInfo]:
        return [ExperimentInfo.from_dict(exp) for exp in self.experiments.values()]
    
    def get_latest_checkpoint(self, experiment_name: str) -> CheckpointInfo | None:
        experiment_info = self.get_experiment_info(experiment_name)
        key_manager.set_key(experiment_info.api_key_name)
        return self.checkpoint_manager.get_checkpoint(experiment_info.job_id)
    
    def delete_experiment(self, experiment_name: str) -> None:
        del self.experiments[experiment_name]
        self._save_experiments()