from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from openai.types import FileObject
from openai.types.fine_tuning.fine_tuning_job import FineTuningJob
from openai.types.fine_tuning.jobs.fine_tuning_job_checkpoint import FineTuningJobCheckpoint

# Define aliases for OpenAI types
FileInfo = FileObject
JobInfo = FineTuningJob
CheckpointInfo = FineTuningJobCheckpoint

@dataclass
class ExperimentInfo:
    name: str
    dataset_id: str
    base_model: str
    file_id: str
    job_id: str
    hyperparameters: Optional[Dict[str, Any]] = None
    api_key_name: str = "default"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentInfo":
        """Create an ExperimentInfo instance from an API response dictionary."""
        return cls(
            name=data["name"],
            dataset_id=data["dataset_id"],
            base_model=data["base_model"],
            file_id=data["file_id"],
            job_id=data["job_id"],
            hyperparameters=data.get("hyperparameters"),
            api_key_name=data.get("api_key_name", "default")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ExperimentInfo instance to a dictionary."""
        return asdict(self)