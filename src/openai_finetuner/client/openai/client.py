"""OpenAI API client implementation."""

from typing import Optional, Dict, Any
import pathlib

from openai import OpenAI
from openai.types import FileObject
from openai.types.fine_tuning.fine_tuning_job import FineTuningJob
from openai.types.fine_tuning.jobs.fine_tuning_job_checkpoint import FineTuningJobCheckpoint

from ...core.interfaces import ClientInterface, Purpose
from .key import get_active_key
from .validate import validate_chat_format

class OpenAIClient(ClientInterface):
    """Direct OpenAI API client without caching."""

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or get_active_key()
        if not api_key:
            raise ValueError("No API key provided")
        self.client = OpenAI(api_key=api_key)

    def create_file(
        self,
        file: str | bytes | pathlib.Path,
        purpose: Purpose = "fine-tune"
    ) -> FileObject:
        """Upload a file to OpenAI API."""
        if purpose == "fine-tune":
            validate_chat_format(file)
        return self.client.files.create(purpose=purpose, file=file)

    def get_file(self, file_id: str) -> FileObject:
        """Retrieve file information from OpenAI."""
        return self.client.files.retrieve(file_id)

    def create_job(
        self,
        file_id: str,
        model: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None
    ) -> FineTuningJob:
        """Create a new fine-tuning job."""
        create_args = {
            "training_file": file_id,
            "model": model
        }
        if hyperparameters:
            create_args["hyperparameters"] = hyperparameters
        if suffix:
            create_args["suffix"] = suffix

        return self.client.fine_tuning.jobs.create(**create_args)

    def get_job(self, job_id: str) -> FineTuningJob:
        """Retrieve job information from OpenAI."""
        return self.client.fine_tuning.jobs.retrieve(job_id)

    def get_checkpoint(self, job_id: str) -> FineTuningJobCheckpoint | None:
        """Get the latest checkpoint for a job."""
        checkpoints = self.list_checkpoints(job_id)
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: x.step_number)

    def list_checkpoints(self, job_id: str) -> list[FineTuningJobCheckpoint]:
        """List all checkpoints for a job."""
        checkpoints = self.client.fine_tuning.jobs.checkpoints.list(job_id)
        return checkpoints.data 