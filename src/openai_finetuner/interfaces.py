import abc
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class FileInfo:
    file_id: str
    dataset_hash: str

@dataclass
class JobInfo:
    job_id: str
    model: str
    training_file: str
    hyperparameters: Dict[str, Any]
    status: str
    fine_tuned_model: Optional[str] = None

class FileManagerInterface(abc.ABC):
    @abc.abstractmethod
    def upload_dataset(self, dataset_id: str) -> FileInfo:
        pass

class JobManagerInterface(abc.ABC):
    @abc.abstractmethod
    def create_job(self, 
        file_id: str, 
        model: str, 
        hyperparameters: Optional[Dict[str, Any]] = None, 
        suffix: Optional[str] = None
    ) -> JobInfo:
        pass