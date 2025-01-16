""" Validate the format of the dataset. """

from collections import defaultdict

from .types import Example

class InvalidDatasetFormatError(Exception):
    """Raised when the dataset format is invalid for fine-tuning."""
    pass


class DatasetValidator: 
    def __init__(self, strategy: str = "chat"):
        self.strategy = strategy
    
    def validate(self, dataset: list[Example]):
        if self.strategy == "chat":
            error_msg = validate_chat_format(dataset)
            if error_msg:
                raise InvalidDatasetFormatError(error_msg)
        elif self.strategy == "completions":
            raise NotImplementedError("Completions format is not implemented yet.")
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        
    # Syntactic sugar for calling the validate method
    def __call__(self, dataset: list[Example]):
        return self.validate(dataset)

def validate_chat_format(dataset: list[Example]):
    """Ensure that the dataset is in the OpenAI format.

    This ensures that the dataset can be used for fine-tuning with the OpenAI API.
    Reference: https://cookbook.openai.com/examples/chat_finetuning_data_prep
    """
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
    
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(
                k not in ("role", "content", "name", "function_call", "weight")
                for k in message
            ):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        error_msg = "Found errors:\n"
        for k, v in format_errors.items():
            error_msg += f"{k}: {v}\n"
        return error_msg.rstrip()
    else:
        return "No errors found"