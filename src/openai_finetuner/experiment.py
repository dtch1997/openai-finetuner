"""Manages experiments that coordinate dataset, file, job and model management."""


from typing import Optional, Dict, Any

from .file import FileManager
from .job import JobManager
from .model import ModelRegistry, ModelInfo
from .key import KeyManager
from .interfaces import FileManagerInterface, JobManagerInterface

key_manager = KeyManager()
model_registry = ModelRegistry()

class ExperimentManager:
    def __init__(
        self,
        file_manager: Optional[FileManagerInterface] = None,
        job_manager: Optional[JobManagerInterface] = None,
    ):
        self.file_manager = file_manager or FileManager(api_key=key_manager.get_active_key())
        self.job_manager = job_manager or JobManager(api_key=key_manager.get_active_key())

    def run_experiment(
        self,
        dataset_id: str,
        base_model: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        experiment_name: str | None = None,
    ) -> ModelInfo:
        """
        Run a complete fine-tuning experiment.
        
        Args:
            dataset_id: ID of dataset to use for training
            base_model: Base model to fine-tune
            hyperparameters: Optional hyperparameters for fine-tuning
            suffix: Optional suffix for the fine-tuned model name
            
        Returns:
            ModelInfo containing details about the fine-tuned model
        """
        # TODO: Verify dataset exists

        # Upload dataset file
        file_info = self.file_manager.upload_dataset(dataset_id)

        # Create fine-tuning job
        job_info = self.job_manager.create_job(
            file_id=file_info.id,
            model=base_model,
            hyperparameters=hyperparameters,
            suffix=experiment_name
        )

        # Register the model
        model_info = ModelInfo(
            model_id=job_info.fine_tuned_model or job_info.job_id,
            base_model=base_model,
            file_id=file_info.id,
            job_id=job_info.id,
            hyperparameters=job_info.hyperparameters
        )
        model_registry.register_model(model_info)

        return model_info
