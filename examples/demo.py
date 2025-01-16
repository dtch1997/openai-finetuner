from openai_finetuner.experiment import ExperimentManager

experiment = ExperimentManager()

experiment.run_experiment(
    dataset_id="my_dataset",
    base_model="gpt-4o-mini",
    hyperparameters={"learning_rate": 0.001},
    suffix="my_model"
)
