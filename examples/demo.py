from openai_finetuner.dataset_registry import DatasetRegistry
from openai_finetuner.file_manager import FileManager
from openai_finetuner.job_manager import JobManager

registry = DatasetRegistry()

# create a dataset
registry.save_dataset([{"messages": [{"role": "user", "content": "Hello, how are you?"}]}], "my_dataset")
print(registry.get_hash("my_dataset"))
print(registry.load_dataset("my_dataset"))

# upload a dataset
file_manager = FileManager()
file_manager.upload_dataset("my_dataset")

# create a job
job_manager = JobManager()
job_manager.create_job(training_file="my_dataset", model="gpt-4o-mini")