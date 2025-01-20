"""Utility functions for computing hashes of files and configurations."""

import hashlib
import json
import pathlib
from typing import Union, Any
import logging

logger = logging.getLogger(__name__)

def compute_file_hash(file: Union[str, bytes, pathlib.Path]) -> str:
    """Compute a SHA-256 hash of file contents for deduplication.
    
    This hash is used to detect duplicate file uploads and avoid unnecessary API calls.
    Files with identical contents will produce the same hash.
    
    Args:
        file: Input file as path, bytes or string
        
    Returns:
        str: Hex digest of SHA-256 hash
        
    Note:
        For large files, this streams the content in chunks to avoid memory issues
    """
    logger.debug(f"Computing file hash for: {file}")
    hasher = hashlib.sha256()
    chunk_size = 8192  # 8KB chunks
    
    if isinstance(file, pathlib.Path):
        with open(file, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
    elif isinstance(file, bytes):
        hasher.update(file)
    elif isinstance(file, str):
        hasher.update(file.encode())
    else:
        raise TypeError(f"Unsupported file type: {type(file)}")
        
    hash_result = hasher.hexdigest()
    logger.debug(f"Computed file hash: {hash_result}")
    return hash_result

def compute_config_hash(**kwargs: Any) -> str:
    """Compute a deterministic hash of configuration for deduplication.
    
    This creates a stable hash of the configuration parameters to detect duplicates.
    Configurations with identical parameters will produce the same hash.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        str: Hex digest of SHA-256 hash
        
    Note:
        - Handles nested dictionaries and lists
        - Order-independent for dictionaries
        - Order-dependent for lists
    """
    logger.debug(f"Computing config hash for: {kwargs}")
    
    def normalize(obj):
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [normalize(x) for x in obj]
        return obj
    
    normalized = normalize(kwargs)
    config_str = json.dumps(normalized, sort_keys=True)
    hash_result = hashlib.sha256(config_str.encode()).hexdigest()
    logger.debug(f"Computed config hash: {hash_result}")
    return hash_result
