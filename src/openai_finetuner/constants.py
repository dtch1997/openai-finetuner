"""Utility functions for managing cache directories."""

import os
import pathlib

_CACHE_DIR_ENV_VAR = "OPENAI_FINETUNER_CACHE_DIR"
_CACHE_DIR_DEFAULT = pathlib.Path.home() / ".cache" / "openai-finetuner"

def get_cache_dir() -> pathlib.Path:
    """Get the cache directory for OpenAI Finetuner.
    
    Returns:
        Path to the cache directory, either from OPENAI_FINETUNER_CACHE_DIR 
        environment variable or ~/.cache/openai-finetuner by default.
    """
    cache_dir = os.getenv(_CACHE_DIR_ENV_VAR, _CACHE_DIR_DEFAULT)
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def set_cache_dir(cache_dir: pathlib.Path):
    """Set the cache directory for OpenAI Finetuner."""
    os.environ[_CACHE_DIR_ENV_VAR] = str(cache_dir)

def clear_cache_dir(cache_dir: pathlib.Path | None = None):
    """Clear the cache directory for OpenAI Finetuner."""
    cache_dir = cache_dir or get_cache_dir()
    for item in cache_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
