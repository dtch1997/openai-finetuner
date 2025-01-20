
from pathlib import Path
from dotenv import load_dotenv
from openai_finetuner.experiment import ExperimentManager
from openai_finetuner.dataset import DatasetManager

curr_dir = Path(__file__).parent
project_dir = curr_dir.parents[1]
load_dotenv(project_dir / ".env")

dataset_manager = DatasetManager()

# Define dataset
dataset = [
    {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm fine, thank you!"}
        ]
    } for _ in range(10)
]

# Register dataset 
dataset_manager.create_dataset(
    id="my_dataset",
    dataset_or_file=dataset
)
print(dataset_manager.list_datasets())

# Create experiment
runner = ExperimentManager()
runner.create_experiment(
    dataset_id="my_dataset",
    base_model="gpt-4o-mini-2024-07-18",
    name="my_experiment",
)

print(runner.list_experiments())
print(runner.get_experiment_info("my_experiment"))

# NOTE: this will fail if the checkpoint is not ready
print(runner.get_latest_checkpoint("my_experiment"))

