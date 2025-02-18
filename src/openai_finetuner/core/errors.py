from openai import RateLimitError as OpenAIRateLimitError

class ExperimentExistsError(ValueError):
    """Raised when attempting to create an experiment with a name that already exists."""
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        super().__init__(f"Experiment {experiment_name} already exists")

class RateLimitError(Exception):
    """Wrapper for OpenAI's RateLimitError that provides a cleaner error message."""
    def __init__(self, original_error: OpenAIRateLimitError):
        self.original_error = original_error
        # Extract the message from the error response
        message = original_error.body.get('error', {}).get('message', str(original_error))
        super().__init__(message) 