from openai_finetuner.core.types import ExperimentInfo

def test_experiment_info():    
    data = {
        "name": "test_experiment",
        "dataset_id": "dataset-123",
        "base_model": "gpt-3.5-turbo",
        "file_id": "file-123",
        "job_id": "job-123",
        "hyperparameters": {"learning_rate": 0.001},
        "api_key_name": "default"
    }
    
    experiment_info = ExperimentInfo.from_dict(data)
    assert isinstance(experiment_info, ExperimentInfo)
    assert experiment_info.name == "test_experiment"
    assert experiment_info.dataset_id == "dataset-123"
    assert experiment_info.base_model == "gpt-3.5-turbo"
    assert experiment_info.api_key_name == "default"
    dict_data = experiment_info.to_dict()
    assert dict_data == data
