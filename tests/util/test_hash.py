"""Unit tests for hash utility functions."""

import pytest
from openai_finetuner.util.hash import compute_file_hash, compute_config_hash

@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file with known content."""
    file_path = tmp_path / "test.txt"
    content = "Hello, World!"
    file_path.write_text(content)
    return file_path, content

def test_calculate_file_hash_from_path(temp_file):
    """Test calculating hash from a file path."""
    file_path, content = temp_file
    hash1 = compute_file_hash(file_path)
    
    # Hash should be consistent
    hash2 = compute_file_hash(file_path)
    assert hash1 == hash2
    
    # Should match hash of same content as string
    str_hash = compute_file_hash(content)
    assert hash1 == str_hash

def test_calculate_file_hash_from_string():
    """Test calculating hash from a string."""
    content = "Test content"
    hash1 = compute_file_hash(content)
    hash2 = compute_file_hash(content)
    assert hash1 == hash2
    
    # Different content should have different hash
    different_hash = compute_file_hash("Different content")
    assert hash1 != different_hash

def test_calculate_file_hash_from_bytes():
    """Test calculating hash from bytes."""
    content = b"Binary content"
    hash1 = compute_file_hash(content)
    hash2 = compute_file_hash(content)
    assert hash1 == hash2

def test_calculate_file_hash_invalid_type():
    """Test calculating hash with invalid input type."""
    with pytest.raises(TypeError, match="Unsupported file type"):
        compute_file_hash(123)

def test_compute_config_hash_simple():
    """Test computing hash for simple configuration."""
    hash1 = compute_config_hash(name="test", value=123)
    hash2 = compute_config_hash(name="test", value=123)
    assert hash1 == hash2
    
    # Different values should have different hashes
    different_hash = compute_config_hash(name="test", value=456)
    assert hash1 != different_hash

def test_compute_config_hash_nested():
    """Test computing hash for nested configuration."""
    config1 = compute_config_hash(
        params={
            "learning_rate": 0.001,
            "batch_size": 32
        },
        model="gpt-3.5-turbo"
    )
    
    config2 = compute_config_hash(
        params={
            "batch_size": 32,
            "learning_rate": 0.001  # Same values, different order
        },
        model="gpt-3.5-turbo"
    )
    
    assert config1 == config2  # Order shouldn't matter for dicts

def test_compute_config_hash_with_lists():
    """Test computing hash with list values."""
    hash1 = compute_config_hash(
        items=["a", "b", "c"],
        value=1
    )
    
    # Same list order should produce same hash
    hash2 = compute_config_hash(
        items=["a", "b", "c"],
        value=1
    )
    assert hash1 == hash2
    
    # Different list order should produce different hash
    different_hash = compute_config_hash(
        items=["c", "b", "a"],
        value=1
    )
    assert hash1 != different_hash

def test_compute_config_hash_complex():
    """Test computing hash for complex nested structure."""
    config = {
        "model": "gpt-3.5-turbo",
        "params": {
            "learning_rate": 0.001,
            "layers": [64, 32, 16],
            "activation": {
                "hidden": "relu",
                "output": "softmax"
            }
        },
        "data": ["train.txt", "test.txt"]
    }
    
    hash1 = compute_config_hash(**config)
    hash2 = compute_config_hash(**config)
    assert hash1 == hash2

def test_compute_config_hash_empty():
    """Test computing hash with no arguments."""
    hash1 = compute_config_hash()
    hash2 = compute_config_hash()
    assert hash1 == hash2
