
from pathlib import Path
from dotenv import load_dotenv
from openai_finetuner.experiment import ExperimentManager
from openai_finetuner.dataset import DatasetManager

curr_dir = Path(__file__).parent
project_dir = curr_dir.parent
load_dotenv(project_dir / ".env")

dataset_manager = DatasetManager()

dataset = {
    "messages": [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"}
    ]
}
dataset_manager.create_dataset(
    id="my_dataset",
    file=dataset
)

runner = ExperimentManager()

runner.create_experiment(
    name="my_experiment",
    dataset_id="my_dataset",
    base_model="gpt-4o-mini",
    hyperparameters={"learning_rate": 0.001},
    suffix="my_model"
)
