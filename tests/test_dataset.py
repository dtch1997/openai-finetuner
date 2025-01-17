import json
import pathlib
import pytest
from openai_finetuner.dataset import DatasetManager, Dataset

@pytest.fixture
def dataset_manager(tmp_path: pathlib.Path) -> DatasetManager:
    return DatasetManager(base_dir=tmp_path)

@pytest.fixture
def sample_dataset() -> Dataset:
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks for asking!"}
            ]
        }
    ]

def test_create_dataset_from_list(dataset_manager: DatasetManager, sample_dataset: Dataset) -> None:
    dataset_id = "test_dataset"
    path = dataset_manager.create_dataset(dataset_id, sample_dataset)
    
    assert path.exists()
    assert path.suffix == ".jsonl"
    
    # Verify contents
    with open(path) as f:
        loaded_data = [json.loads(line) for line in f]
    assert loaded_data == sample_dataset

def test_create_dataset_from_json_str(dataset_manager: DatasetManager, sample_dataset: Dataset) -> None:
    dataset_id = "test_dataset"
    json_str = json.dumps(sample_dataset).encode()
    path = dataset_manager.create_dataset(dataset_id, json_str)
    
    with open(path) as f:
        loaded_data = [json.loads(line) for line in f]
    assert loaded_data == sample_dataset

def test_retrieve_dataset(dataset_manager: DatasetManager, sample_dataset: Dataset) -> None:
    dataset_id = "test_dataset"
    dataset_manager.create_dataset(dataset_id, sample_dataset)
    
    loaded_dataset = dataset_manager.retrieve_dataset(dataset_id)
    assert loaded_dataset == sample_dataset

def test_get_dataset_path(dataset_manager: DatasetManager) -> None:
    dataset_id = "test_dataset"
    expected_path = dataset_manager.datasets_dir / "test_dataset.jsonl"
    assert dataset_manager.get_dataset_path(dataset_id) == expected_path

def test_remove_dataset(dataset_manager: DatasetManager, sample_dataset: Dataset) -> None:
    dataset_id = "test_dataset"
    path = dataset_manager.create_dataset(dataset_id, sample_dataset)
    assert path.exists()
    
    dataset_manager.remove_dataset(dataset_id)
    assert not path.exists()

def test_retrieve_nonexistent_dataset(dataset_manager: DatasetManager) -> None:
    with pytest.raises(FileNotFoundError):
        dataset_manager.retrieve_dataset("nonexistent")

def test_create_dataset_invalid_input(dataset_manager: DatasetManager) -> None:
    with pytest.raises(ValueError):
        dataset_manager.create_dataset("test", 123)  # type: ignore

def test_list_datasets(dataset_manager: DatasetManager, sample_dataset: Dataset) -> None:
    # Create multiple datasets
    dataset_ids = ["test1", "test2", "test3"]
    for dataset_id in dataset_ids:
        dataset_manager.create_dataset(dataset_id, sample_dataset)
    
    # List datasets and verify
    listed_datasets = dataset_manager.list_datasets()
    assert sorted(listed_datasets) == sorted(dataset_ids)
