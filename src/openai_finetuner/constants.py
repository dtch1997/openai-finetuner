"""Utility functions for managing cache directories."""

import os
import pathlib

def get_cache_dir() -> pathlib.Path:
    """Get the cache directory for OpenAI Finetuner.
    
    Returns:
        Path to the cache directory, either from OPENAI_FINETUNER_CACHE_DIR 
        environment variable or ~/.cache/openai-finetuner by default.
    """
    cache_dir = os.getenv("OPENAI_FINETUNER_CACHE_DIR")
    if cache_dir is None:
        cache_dir = pathlib.Path.home() / ".cache" / "openai-finetuner"
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

