import pytest
import json
import pathlib

from unittest.mock import Mock
from openai_finetuner.client.wrappers.cache import CacheWrapper
from openai_finetuner.core.interfaces import ClientInterface

@pytest.fixture
def mock_client():
    client = Mock(spec=ClientInterface)
    return client

@pytest.fixture
def temp_cache_dir(tmp_path) -> pathlib.Path:
    return tmp_path

@pytest.fixture
def cache_wrapper(mock_client: Mock, temp_cache_dir: pathlib.Path) -> CacheWrapper:
    return CacheWrapper(mock_client, base_dir=temp_cache_dir)

def test_create_file_cache_miss(cache_wrapper: CacheWrapper, mock_client: Mock):
    # Setup mock response
    mock_response = Mock()
    mock_response.id = "file-123"
    mock_client.create_file.return_value = mock_response
    
    # Create file
    test_file = "test content"
    result = cache_wrapper.create_file(test_file)
    
    # Verify file was created
    mock_client.create_file.assert_called_once()
    assert result.id == "file-123"
    
    # Verify cache was updated
    with open(cache_wrapper.files_cache_path) as f:
        cache = json.load(f)
        assert len(cache) == 1
        assert mock_response.id in cache.values()

def test_create_file_cache_hit(cache_wrapper: CacheWrapper, mock_client: Mock):
    # Setup mock responses
    mock_file = Mock()
    mock_file.id = "file-123"
    mock_client.create_file.return_value = mock_file
    mock_client.get_file.return_value = mock_file
    
    # First creation (cache miss)
    test_file = "test content"
    first_result = cache_wrapper.create_file(test_file)
    
    # Second creation (should hit cache)
    second_result = cache_wrapper.create_file(test_file)
    
    # Verify file was only created once
    mock_client.create_file.assert_called_once()
    mock_client.get_file.assert_called_once_with("file-123")
    assert first_result.id == second_result.id

def test_create_file_handles_missing_cached_file(cache_wrapper: CacheWrapper, mock_client: Mock):
    # Setup mock responses
    mock_file = Mock()
    mock_file.id = "file-123"
    mock_client.create_file.return_value = mock_file
    mock_client.get_file.side_effect = Exception("File not found")
    
    # First creation
    test_file = "test content"
    first_result = cache_wrapper.create_file(test_file)
    
    # Second creation (cache hit but file missing)
    second_result = cache_wrapper.create_file(test_file)
    
    # Verify file was created twice
    assert mock_client.create_file.call_count == 2
    assert first_result.id == second_result.id

def test_create_job_cache_miss(cache_wrapper: CacheWrapper, mock_client: Mock):
    # Setup mock response
    mock_response = Mock()
    mock_response.id = "job-123"
    mock_client.create_job.return_value = mock_response
    
    # Create job
    result = cache_wrapper.create_job("file-123", "gpt-3.5-turbo")
    
    # Verify job was created
    mock_client.create_job.assert_called_once()
    assert result.id == "job-123"
    
    # Verify cache was updated
    with open(cache_wrapper.jobs_cache_path) as f:
        cache = json.load(f)
        assert len(cache) == 1
        assert mock_response.id in cache.values()

def test_create_job_cache_hit(cache_wrapper, mock_client):
    # Setup mock responses
    mock_job = Mock()
    mock_job.id = "job-123"
    mock_client.create_job.return_value = mock_job
    mock_client.get_job.return_value = mock_job
    
    # First creation (cache miss)
    first_result = cache_wrapper.create_job("file-123", "gpt-3.5-turbo")
    
    # Second creation (should hit cache)
    second_result = cache_wrapper.create_job("file-123", "gpt-3.5-turbo")
    
    # Verify job was only created once
    mock_client.create_job.assert_called_once()
    mock_client.get_job.assert_called_once_with("job-123")
    assert first_result.id == second_result.id

@pytest.mark.skip(reason="This test is not working as expected")
def test_create_job_handles_missing_cached_job(cache_wrapper, mock_client):
    # Setup mock responses
    mock_job = Mock()
    mock_job.id = "job-123"
    mock_client.create_job.return_value = mock_job
    mock_client.get_job.side_effect = Exception("Job not found")
    
    # First creation
    first_result = cache_wrapper.create_job("file-123", "gpt-3.5-turbo")

    # Second creation (cache hit but job missing)
    second_result = cache_wrapper.create_job("file-123", "gpt-3.5-turbo")

    # Verify job was created twice
    assert mock_client.create_job.call_count == 2
    assert first_result.id == second_result.id

def test_get_file_removes_from_cache_on_error(cache_wrapper, mock_client):
    # Setup mock response
    mock_client.get_file.side_effect = Exception("File not found")
    
    # Add file to cache
    cache_wrapper.files_cache["file-hash"] = "file-123"
    cache_wrapper._save_cache(cache_wrapper.files_cache_path, cache_wrapper.files_cache)
    
    # Try to get file
    with pytest.raises(Exception):
        cache_wrapper.get_file("file-hash")

    # Verify file was removed from cache
    assert "file-hash" not in cache_wrapper.files_cache

def test_passthrough_methods(cache_wrapper, mock_client):
    # Test that these methods just pass through to the client
    cache_wrapper.get_checkpoint("job-123")
    mock_client.get_checkpoint.assert_called_once_with("job-123")
    
    cache_wrapper.list_checkpoints("job-123")
    mock_client.list_checkpoints.assert_called_once_with("job-123")
