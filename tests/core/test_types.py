from openai_finetuner.core.types import FileInfo, JobInfo, CheckpointInfo, ExperimentInfo

def test_checkpoint_info():
    data = {
        "id": "checkpoint-123",
        "object": "fine_tuning.job.checkpoint",
        "created_at": 1623456789,
        "fine_tuned_model_checkpoint": "checkpoint-1",
        "fine_tuning_job_id": "job-123",
        "metrics": {"loss": 0.5},
        "step_number": 100
    }
    
    checkpoint_info = CheckpointInfo.from_dict(data)
    assert isinstance(checkpoint_info, CheckpointInfo)
    assert checkpoint_info.id == "checkpoint-123"
    assert checkpoint_info.object == "fine_tuning.job.checkpoint"
    assert checkpoint_info.metrics == {"loss": 0.5}
    
    dict_data = checkpoint_info.to_dict()
    assert dict_data == data

def test_experiment_info():
    file_data = {
        "id": "file-123",
        "object": "file",
        "bytes": 1000,
        "created_at": 1623456789,
        "filename": "test.jsonl",
        "purpose": "fine-tune",
        "hash": "abc123"
    }
    
    job_data = {
        "id": "job-123",
        "object": "fine_tuning.job",
        "model": "gpt-3.5-turbo",
        "created_at": 1623456789,
        "finished_at": 1623457789,
        "fine_tuned_model": "ft:gpt-3.5-turbo:org:custom_suffix:id",
        "organization_id": "org-123",
        "result_files": ["file-456"],
        "status": "succeeded",
        "validation_file": None,
        "training_file": "file-789",
        "hyperparameters": {"n_epochs": 3},
        "trained_tokens": 1000,
        "integrations": [],
        "seed": 42,
        "estimated_finish": 1623458789,
        "method": {"type": "standard"}
    }
    
    data = {
        "name": "test_experiment",
        "dataset_id": "dataset-123",
        "base_model": "gpt-3.5-turbo",
        "file_info": file_data,
        "job_info": job_data,
        "hyperparameters": {"learning_rate": 0.001}
    }
    
    experiment_info = ExperimentInfo.from_dict(data)
    assert isinstance(experiment_info, ExperimentInfo)
    assert experiment_info.name == "test_experiment"
    assert experiment_info.dataset_id == "dataset-123"
    assert experiment_info.base_model == "gpt-3.5-turbo"
    assert isinstance(experiment_info.file_info, FileInfo)
    assert isinstance(experiment_info.job_info, JobInfo)
    
    dict_data = experiment_info.to_dict()
    assert dict_data == data
