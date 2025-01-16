"""Manages fine-tuned models."""
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .constants import get_cache_dir

@dataclass
class ModelInfo:
    model_id: str
    base_model: str
    dataset_id: str
    job_id: str
    hyperparameters: Dict[str, Any]

class ModelRegistry:
    def __init__(self, base_dir: pathlib.Path = get_cache_dir() / ".models"):
        self.base_dir = pathlib.Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.models_file = self.base_dir / "models.json"
        self._load_models()

    def _load_models(self):
        """Load the models registry from disk."""
        if self.models_file.exists():
            with open(self.models_file) as f:
                self.models = json.load(f)
        else:
            self.models = {}

    def _save_models(self):
        """Save the models registry to disk."""
        temp_file = self.models_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.models, f, indent=2)
        os.replace(temp_file, self.models_file)

    def register_model(self, model_info: ModelInfo) -> None:
        """Register a new fine-tuned model."""
        self.models[model_info.model_id] = {
            "base_model": model_info.base_model,
            "dataset_id": model_info.dataset_id,
            "job_id": model_info.job_id,
            "hyperparameters": model_info.hyperparameters
        }
        self._save_models()

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a registered model."""
        if model_id not in self.models:
            return None
        model = self.models[model_id]
        return ModelInfo(
            model_id=model_id,
            base_model=model["base_model"],
            dataset_id=model["dataset_id"],
            job_id=model["job_id"],
            hyperparameters=model["hyperparameters"]
        )