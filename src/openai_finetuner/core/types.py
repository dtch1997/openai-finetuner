from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from typing_extensions import Literal

@dataclass(frozen=True)
class FileInfo:
    id: str
    object: Literal["file"]
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the FileInfo instance to a dictionary."""
        return asdict(self)

# TODO: Can't make this frozen because some fields are mutable
@dataclass
class JobInfo:
    id: str
    object: Literal["fine_tuning.job"]
    model: str
    created_at: int
    finished_at: int
    fine_tuned_model: str
    organization_id: str
    result_files: List[str]
    status: str
    validation_file: str | None
    training_file: str
    hyperparameters: Dict[str, Any]
    trained_tokens: int
    integrations: list[str]
    seed: int
    estimated_finish: int
    method: dict[str, Any]

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the JobInfo instance to a dictionary."""
        return asdict(self)

@dataclass
class CheckpointInfo:
    id: str
    object: Literal["fine_tuning.job.checkpoint"]
    created_at: int
    fine_tuned_model_checkpoint: str
    fine_tuning_job_id: str
    metrics: Dict[str, float]
    step_number: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointInfo":
        """Create a CheckpointInfo instance from an API response dictionary."""
        return cls(
            id=data["id"],
            object=data["object"],
            created_at=data["created_at"],
            fine_tuned_model_checkpoint=data["fine_tuned_model_checkpoint"],
            fine_tuning_job_id=data["fine_tuning_job_id"],
            metrics=data["metrics"],
            step_number=data["step_number"]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the CheckpointInfo instance to a dictionary."""
        return asdict(self)

@dataclass
class ExperimentInfo:
    name: str
    dataset_id: str
    base_model: str
    file_info: FileInfo
    job_info: JobInfo
    hyperparameters: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentInfo":
        """Create an ExperimentInfo instance from an API response dictionary."""
        return cls(
            name=data["name"],
            dataset_id=data["dataset_id"],
            base_model=data["base_model"],
            file_info=FileInfo.from_dict(data["file_info"]),
            job_info=JobInfo.from_dict(data["job_info"]),
            hyperparameters=data.get("hyperparameters")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ExperimentInfo instance to a dictionary."""
        return asdict(self)
