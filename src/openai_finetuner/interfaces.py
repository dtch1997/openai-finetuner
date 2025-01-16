import abc
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass(frozen=True)
class FileInfo:
    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    hash: str # hash of the file contents

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileInfo":
        """Create a FileInfo instance from an API response dictionary."""
        return cls(
            id=data["id"],
            object=data["object"],
            bytes=data["bytes"],
            created_at=data["created_at"],
            filename=data["filename"],
            purpose=data["purpose"],
            hash=data["hash"]
        )

@dataclass(frozen=True)
class JobInfo:
    id: str
    object: str = "fine_tuning.job"
    model: str
    created_at: int
    finished_at: Optional[int] = None
    fine_tuned_model: Optional[str] = None
    organization_id: str
    result_files: List[str]
    status: str
    validation_file: Optional[str] = None
    training_file: str
    hyperparameters: Dict[str, Any]
    trained_tokens: Optional[int] = None
    integrations: List[str] = None
    seed: int = 0
    estimated_finish: int = 0
    method: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobInfo":
        """Create a JobInfo instance from an API response dictionary."""
        return cls(
            id=data["id"],
            object=data["object"],
            model=data["model"],
            created_at=data["created_at"],
            finished_at=data.get("finished_at"),
            fine_tuned_model=data.get("fine_tuned_model"),
            organization_id=data["organization_id"],
            result_files=data["result_files"],
            status=data["status"],
            validation_file=data.get("validation_file"),
            training_file=data["training_file"],
            hyperparameters=data["hyperparameters"],
            trained_tokens=data.get("trained_tokens"),
            integrations=data.get("integrations", []),
            seed=data.get("seed", 0),
            estimated_finish=data.get("estimated_finish", 0),
            method=data.get("method")
        )

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