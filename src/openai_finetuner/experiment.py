"""Manages experiments that coordinate dataset, file, job and model management."""


from typing import Optional, Dict, Any

from .dataset import DatasetRegistry
from .file import FileManager
from .job import JobManager
from .model import ModelRegistry, ModelInfo
from .key import KeyManager

class ExperimentManager:
    def __init__(
        self,
        key_manager: Optional[KeyManager] = None,
        dataset_registry: Optional[DatasetRegistry] = None,
        file_manager: Optional[FileManager] = None,
        job_manager: Optional[JobManager] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        self.key_manager = key_manager or KeyManager()
        self.dataset_registry = dataset_registry or DatasetRegistry()
        self.file_manager = file_manager or FileManager(api_key=self.key_manager.get_active_key())
        self.job_manager = job_manager or JobManager(api_key=self.key_manager.get_active_key())
        self.model_registry = model_registry or ModelRegistry()

    def run_experiment(self,
                      dataset_id: str,
                      base_model: str,
                      hyperparameters: Optional[Dict[str, Any]] = None,
                      suffix: Optional[str] = None) -> ModelInfo:
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
        # Verify dataset exists
        if not self.dataset_registry.get_hash(dataset_id):
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Upload dataset file
        file_info = self.file_manager.upload_dataset(dataset_id)

        # Create fine-tuning job
        job_info = self.job_manager.create_job(
            training_file=file_info.file_id,
            model=base_model,
            hyperparameters=hyperparameters,
            suffix=suffix
        )

        # Register the model
        model_info = ModelInfo(
            model_id=job_info.fine_tuned_model or job_info.job_id,
            base_model=base_model,
            dataset_id=dataset_id,
            job_id=job_info.job_id,
            hyperparameters=job_info.hyperparameters
        )
        self.model_registry.register_model(model_info)

        return model_info
