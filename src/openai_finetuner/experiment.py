"""Manages experiments that coordinate dataset, file, job and model management."""

from typing import Optional, Dict, Any
import json
import pathlib

from .client.openai.client import OpenAIClient
from .client.wrappers.cache import CacheWrapper
from .core.interfaces import (
    ExperimentManagerInterface,
    ClientInterface
)
from .core.types import ( 
    ExperimentInfo, 
    JobInfo, 
    FileInfo,
    CheckpointInfo
)
from .constants import get_cache_dir
from .dataset import DatasetManager

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

class ExperimentManager(ExperimentManagerInterface):
    """Manages fine-tuning experiments and their associated resources.

    The ExperimentManager coordinates the creation and tracking of fine-tuning experiments,
    including dataset files, training jobs, and model checkpoints. It persists experiment
    metadata to disk and provides methods to query experiment status and results.

    Args:
        client: Optional client for API interactions. If not provided,
            a default OpenAIClient will be created.
        base_dir: Optional path for storing experiment data and metadata. Defaults to
            the cache directory.

    Attributes:
        client: The OpenAI client interface used for API calls
        base_dir: Directory where experiment data is stored
        experiments_file: JSON file containing experiment metadata
        dataset_manager: Manager for handling training datasets

    Example:        
    ```python
        manager = ExperimentManager()

        # Ensure 'my_dataset' exists
        
        # Create a new fine-tuning experiment
        experiment = manager.create_experiment(
            dataset_id="my_dataset",
            base_model="gpt-3.5-turbo",
            name="my_experiment"
        )

        # Get status of the experiment
        job_info = manager.get_job_info("my_experiment")        ```
    """

    def __init__(
        self,
        client: Optional[ClientInterface] = None,
        dataset_manager: Optional[DatasetManager] = None,
        base_dir: pathlib.Path = get_cache_dir()
    ):
        self.client = client or CacheWrapper(OpenAIClient())
        self.dataset_manager = dataset_manager or DatasetManager()
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
            raise ValueError(f"Experiment {name} already exists")

        # Upload dataset file
        file_info = self.client.create_file(
            file=self.dataset_manager.get_dataset_path(dataset_id),
        )

        # Create fine-tuning job
        job_info = self.client.create_job(
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
            api_key_name=None  # Remove api_key_name since we're not using KeyManager
        )

        # Save experiment
        self.experiments[name] = experiment_info.to_dict()
        self._save_experiments()

        return experiment_info

    def get_experiment_info(self, experiment_name: str) -> ExperimentInfo:
        return ExperimentInfo.from_dict(self.experiments[experiment_name])

    def get_job_info(self, experiment_name: str) -> JobInfo:
        experiment_info = self.get_experiment_info(experiment_name)
        return self.client.get_job(experiment_info.job_id)

    def get_file_info(self, experiment_name: str) -> FileInfo:
        experiment_info = self.get_experiment_info(experiment_name)
        return self.client.get_file(experiment_info.file_id)
    
    def list_experiments(self) -> list[ExperimentInfo]:
        return [ExperimentInfo.from_dict(exp) for exp in self.experiments.values()]
    
    def get_latest_checkpoint(self, experiment_name: str) -> CheckpointInfo | None:
        experiment_info = self.get_experiment_info(experiment_name)
        return self.client.get_checkpoint(experiment_info.job_id)
    
    def delete_experiment(self, experiment_name: str) -> None:
        del self.experiments[experiment_name]
        self._save_experiments()