import pytest
from pathlib import Path
from unittest.mock import Mock
import os

from openai_finetuner.experiment import ExperimentManager, JobFailedError
from openai_finetuner.core.types import (
    ExperimentInfo,
    JobInfo,
    FileInfo,
    CheckpointInfo
)
from openai_finetuner.constants import _CACHE_DIR_ENV_VAR
from openai_finetuner.core.errors import ExperimentExistsError

@pytest.fixture
def mock_client():
    client = Mock()
    # Setup default return values
    client.create_file.return_value = FileInfo(
        id="file-123",
        bytes=1000,
        filename="test.jsonl",
        created_at=1716150000,
        purpose="fine-tune",
        object="file",
        status="uploaded"
    )
    client.create_job.return_value = JobInfo(
        id="job-123",
        status="running",
        model="gpt-3.5-turbo",
        error=None,
        created_at=1716150000,
        object="fine_tuning.job",
        organization_id="org-123",
        result_files=[],
        training_file="file-123",
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 1,
            "learning_rate_multiplier": 1.0
        },
        seed=42
    )
    return client

@pytest.fixture
def mock_dataset_manager():
    manager = Mock()
    manager.get_dataset_path.return_value = Path("test_dataset.jsonl")
    return manager

@pytest.fixture
def experiment_manager(mock_client, mock_dataset_manager, tmp_path):
    return ExperimentManager(
        client=mock_client,
        dataset_manager=mock_dataset_manager,
        base_dir=tmp_path
    )

def test_create_experiment(experiment_manager, mock_client):
    """Test creating a new experiment"""
    experiment = experiment_manager.create_experiment(
        dataset_id="test_dataset",
        base_model="gpt-3.5-turbo",
        name="test_experiment"
    )

    assert experiment.name == "test_experiment"
    assert experiment.dataset_id == "test_dataset"
    assert experiment.base_model == "gpt-3.5-turbo"
    assert experiment.file_id == "file-123"
    assert experiment.job_id == "job-123"

    # Verify client calls
    mock_client.create_file.assert_called_once()
    mock_client.create_job.assert_called_once()

def test_create_experiment_duplicate_name(experiment_manager):
    """Test that creating an experiment with duplicate name raises error"""
    experiment_manager.create_experiment(
        dataset_id="test_dataset",
        base_model="gpt-3.5-turbo",
        name="test_experiment"
    )

    with pytest.raises(ExperimentExistsError) as exc_info:
        experiment_manager.create_experiment(
            dataset_id="test_dataset",
            base_model="gpt-3.5-turbo",
            name="test_experiment"
        )
    
    assert exc_info.value.experiment_name == "test_experiment"

def test_create_experiment_failed_job(experiment_manager, mock_client):
    """Test handling of failed job creation"""
    mock_client.create_job.return_value = JobInfo(
        id="job-failed",
        status="failed",
        model="gpt-3.5-turbo",
        error=None,
        created_at=1716150000,
        object="fine_tuning.job",
        organization_id="org-123",
        result_files=[],
        training_file="file-123",
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 1,
            "learning_rate_multiplier": 1.0
        },
        seed=42
    )

    with pytest.raises(JobFailedError) as exc_info:
        experiment_manager.create_experiment(
            dataset_id="test_dataset",
            base_model="gpt-3.5-turbo",
            name="test_experiment"
        )
    
    assert exc_info.value.job_id == "job-failed"

def test_get_experiment_info(experiment_manager):
    """Test retrieving experiment info"""
    # Create an experiment first
    experiment_manager.create_experiment(
        dataset_id="test_dataset",
        base_model="gpt-3.5-turbo",
        name="test_experiment"
    )

    # Get the experiment info
    info = experiment_manager.get_experiment_info("test_experiment")
    assert isinstance(info, ExperimentInfo)
    assert info.name == "test_experiment"
    assert info.dataset_id == "test_dataset"
    assert info.base_model == "gpt-3.5-turbo"

def test_get_job_info(experiment_manager, mock_client):
    """Test retrieving job info"""
    # Create an experiment
    experiment_manager.create_experiment(
        dataset_id="test_dataset",
        base_model="gpt-3.5-turbo",
        name="test_experiment"
    )

    # Setup mock return value for get_job
    mock_client.get_job.return_value = JobInfo(
        id="job-123",
        status="succeeded",
        model="gpt-3.5-turbo",
        error=None,
        created_at=1716150000,
        object="fine_tuning.job",
        organization_id="org-123",
        result_files=[],
        training_file="file-123",
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 1,
            "learning_rate_multiplier": 1.0
        },
        seed=42
    )

    # Get job info
    job_info = experiment_manager.get_job_info("test_experiment")
    assert isinstance(job_info, JobInfo)
    assert job_info.id == "job-123"
    assert job_info.status == "succeeded"

def test_list_experiments(experiment_manager):
    """Test listing all experiments"""
    # Create multiple experiments
    experiment_manager.create_experiment(
        dataset_id="test_dataset1",
        base_model="gpt-3.5-turbo",
        name="test_experiment1"
    )
    experiment_manager.create_experiment(
        dataset_id="test_dataset2",
        base_model="gpt-3.5-turbo",
        name="test_experiment2"
    )

    experiments = experiment_manager.list_experiments()
    assert len(experiments) == 2
    assert all(isinstance(exp, ExperimentInfo) for exp in experiments)
    assert {exp.name for exp in experiments} == {"test_experiment1", "test_experiment2"}

def test_delete_experiment(experiment_manager):
    """Test deleting an experiment"""
    # Create an experiment
    experiment_manager.create_experiment(
        dataset_id="test_dataset",
        base_model="gpt-3.5-turbo",
        name="test_experiment"
    )

    # Delete the experiment
    experiment_manager.delete_experiment("test_experiment")

    # Verify it's gone
    with pytest.raises(KeyError):
        experiment_manager.get_experiment_info("test_experiment")

def test_persistence(mock_client, mock_dataset_manager, tmp_path):
    """Test that experiments persist between manager instances"""
    # Create first manager instance
    manager1 = ExperimentManager(
        client=mock_client,
        dataset_manager=mock_dataset_manager,
        base_dir=tmp_path
    )
    
    # Create an experiment
    manager1.create_experiment(
        dataset_id="test_dataset",
        base_model="gpt-3.5-turbo",
        name="test_experiment"
    )

    # Create second manager instance
    manager2 = ExperimentManager(
        client=mock_client,
        dataset_manager=mock_dataset_manager,
        base_dir=tmp_path
    )

    # Verify experiment exists in second instance
    experiment = manager2.get_experiment_info("test_experiment")
    assert experiment.name == "test_experiment"
    assert experiment.dataset_id == "test_dataset"

def test_get_latest_checkpoint(experiment_manager, mock_client):
    """Test retrieving the latest checkpoint"""
    # Create an experiment
    experiment_manager.create_experiment(
        dataset_id="test_dataset",
        base_model="gpt-3.5-turbo",
        name="test_experiment"
    )

    # Setup mock return value for get_checkpoint
    mock_client.get_checkpoint.return_value = CheckpointInfo(
        id="checkpoint-123",
        model="ft:gpt-3.5-turbo:org:test:123",
        created_at=1716150000,
        fine_tuned_model_checkpoint="checkpoint-123",
        fine_tuning_job_id="job-123",
        object="fine_tuning.job.checkpoint",
        step_number=100,
        metrics={
            "train_loss": 0.5,
            "test_loss": 0.6
        }
    )

    # Get checkpoint info
    checkpoint = experiment_manager.get_latest_checkpoint("test_experiment")
    assert isinstance(checkpoint, CheckpointInfo)
    assert checkpoint.id == "checkpoint-123"
    assert checkpoint.model == "ft:gpt-3.5-turbo:org:test:123"

def test_experiment_manager_uses_env_cache_dir(mock_client, mock_dataset_manager, tmp_path):
    """Test that ExperimentManager uses the cache directory from environment variable."""
    custom_cache_dir = tmp_path / "custom_cache"
    os.environ[_CACHE_DIR_ENV_VAR] = str(custom_cache_dir)
    
    try:
        # Create manager without explicit base_dir
        manager = ExperimentManager(
            client=mock_client,
            dataset_manager=mock_dataset_manager
        )
        
        # Verify it uses the custom cache dir
        assert manager.base_dir == custom_cache_dir
        assert manager.experiments_file == custom_cache_dir / "experiments.json"
        
        # Test functionality with custom cache dir
        experiment = manager.create_experiment(
            dataset_id="test_dataset",
            base_model="gpt-3.5-turbo",
            name="test_experiment"
        )
        
        # Verify data persistence
        loaded_experiment = manager.get_experiment_info("test_experiment")
        assert loaded_experiment.name == experiment.name
        assert loaded_experiment.dataset_id == experiment.dataset_id
        
        # Verify file was created in correct location
        assert manager.experiments_file.exists()
        
    finally:
        # Clean up environment
        del os.environ[_CACHE_DIR_ENV_VAR]

def test_experiment_manager_env_cache_dir_persistence(mock_client, mock_dataset_manager, tmp_path):
    """Test that experiments persist between manager instances when using env cache dir."""
    custom_cache_dir = tmp_path / "custom_cache"
    os.environ[_CACHE_DIR_ENV_VAR] = str(custom_cache_dir)
    
    try:
        # Create first manager instance
        manager1 = ExperimentManager(
            client=mock_client,
            dataset_manager=mock_dataset_manager
        )
        
        # Create an experiment
        manager1.create_experiment(
            dataset_id="test_dataset",
            base_model="gpt-3.5-turbo",
            name="test_experiment"
        )
        
        # Create second manager instance
        manager2 = ExperimentManager(
            client=mock_client,
            dataset_manager=mock_dataset_manager
        )
        
        # Verify experiment exists in second instance
        experiment = manager2.get_experiment_info("test_experiment")
        assert experiment.name == "test_experiment"
        assert experiment.dataset_id == "test_dataset"
        assert manager2.base_dir == custom_cache_dir
        
    finally:
        # Clean up environment
        del os.environ[_CACHE_DIR_ENV_VAR]
