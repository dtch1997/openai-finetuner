import pytest
from openai import RateLimitError as OpenAIRateLimitError
from openai_finetuner.core.errors import ExperimentExistsError, RateLimitError
import httpx

def test_experiment_exists_error():
    """Test that ExperimentExistsError has correct message and attributes."""
    experiment_name = "test_experiment"
    with pytest.raises(ExperimentExistsError) as exc_info:
        raise ExperimentExistsError(experiment_name)
    
    error = exc_info.value
    assert error.experiment_name == experiment_name
    assert str(error) == f"Experiment {experiment_name} already exists"

def test_rate_limit_error():
    """Test that RateLimitError correctly wraps OpenAI's RateLimitError."""
    original_message = "Rate limit exceeded. Please try again in 20s."
    
    # Create mock request and response
    request = httpx.Request("POST", "https://api.openai.com/v1/files")
    response = httpx.Response(
        status_code=429,
        request=request,
        json={
            "error": {
                "message": original_message,
                "type": "rate_limit_error",
                "code": "rate_limit"
            }
        }
    )
    
    # Create the rate limit error
    original_error = OpenAIRateLimitError(
        message="Rate limit reached",
        body={'error': {'message': original_message}},
        response=response,
    )
    
    with pytest.raises(RateLimitError) as exc_info:
        raise RateLimitError(original_error)
    
    error = exc_info.value
    assert str(error) == original_message
    assert error.original_error == original_error 