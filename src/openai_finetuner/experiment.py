"""Manages experiments that coordinate dataset, file, job and model management."""


from typing import Optional, Dict, Any
import json
import pathlib

from .openai.file import FileManager
from .openai.job import JobManager
from .core.interfaces import FileManagerInterface, JobManagerInterface
from .core.types import ExperimentInfo
from .constants import get_cache_dir
from .dataset import DatasetManager

dataset_manager = DatasetManager()

class ExperimentManager:
    def __init__(
        self,
        file_manager: Optional[FileManagerInterface] = None,
        job_manager: Optional[JobManagerInterface] = None,
        base_dir: pathlib.Path = get_cache_dir()
    ):
        self.file_manager = file_manager or FileManager()
        self.job_manager = job_manager or JobManager()
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

        # Create experiment info
        experiment_info = ExperimentInfo(
            name=name,
            dataset_id=dataset_id,
            base_model=base_model,
            file_info=file_info,
            job_info=job_info,
            hyperparameters=hyperparameters
        )

        # Save experiment
        self.experiments[name] = experiment_info.to_dict()
        self._save_experiments()

        return experiment_info
