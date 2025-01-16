import abc
import pathlib

from typing import Optional, Dict, Any, Literal

from .types import ExperimentInfo, CheckpointInfo, FileInfo, JobInfo

Purpose = Literal["fine-tune", "batch"]

class FileManagerInterface(abc.ABC):
    @abc.abstractmethod
    def create_file(
        self,
        file: str | bytes | pathlib.Path,
        purpose: Purpose = "fine-tune"
    ) -> FileInfo:
        pass

    @abc.abstractmethod
    def get_file(self, file_id: str) -> FileInfo:
        pass

class JobManagerInterface(abc.ABC):
    @abc.abstractmethod
    def create_job(
        self, 
        file_id: str, 
        model: str, 
        hyperparameters: Optional[Dict[str, Any]] = None, 
        suffix: Optional[str] = None
    ) -> JobInfo:
        pass

    @abc.abstractmethod
    def get_job(self, job_id: str) -> JobInfo:
        pass

class ExperimentManagerInterface(abc.ABC):

    file_manager: FileManagerInterface
    job_manager: JobManagerInterface

    @abc.abstractmethod
    def create_experiment(
        self,
        name: str,
        dataset_id: str,
        base_model: str,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> ExperimentInfo:
        """
        Create and run a fine-tuning experiment.

        Args:
            name: Name of the experiment
            dataset_id: ID of dataset to use for training
            base_model: Base model to fine-tune
            hyperparameters: Optional hyperparameters for fine-tuning

        Returns:
            ExperimentInfo containing all details about the experiment
        """
        pass

    @abc.abstractmethod
    def list_experiments(self) -> list[ExperimentInfo]:
        pass

class CheckpointManagerInterface(abc.ABC):
    @abc.abstractmethod
    def get_checkpoint(self, job_id: str) -> CheckpointInfo:
        pass
