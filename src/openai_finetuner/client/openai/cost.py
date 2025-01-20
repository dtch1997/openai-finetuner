""" Estimate the cost of fine-tuning a model. """

import numpy as np
import tiktoken
from dataclasses import dataclass

from .types import Example

encoding = tiktoken.get_encoding("cl100k_base")

# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 16385

TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

# Fine-tuning cost 
# https://openai.com/api/pricing/

MODEL_FINETUNE_COST_USD_PER_1M_TOKENS = {
    "gpt-4o-2024-08-06": 25,
    "gpt-4o-mini-2024-07-18": 3,
    "gpt-3.5-turbo": 8,
}


@dataclass
class DatasetCostInfo:
    n_billing_tokens: int
    n_epochs: int

# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


def estimate_cost_info(dataset: list[Example]) -> DatasetCostInfo:
    """Estimate the cost of training on the dataset."""

    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens = sum(
        min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens
    )
    
    return DatasetCostInfo(
        n_billing_tokens=n_billing_tokens,
        n_epochs=n_epochs,
    )

def get_cost(
    dataset: list[Example],
    model: str,
) -> float:
    cost_info = estimate_cost_info(dataset)
    return cost_info.n_billing_tokens * MODEL_FINETUNE_COST_USD_PER_1M_TOKENS[model] * cost_info.n_epochs / 1e6

def format_cost_estimate(cost_info: DatasetCostInfo) -> str:
    """Format the cost estimate as a human readable string."""
    return (
        f"Dataset has ~{cost_info.n_billing_tokens} tokens that will be charged for during training\n"
        f"By default, you'll train for {cost_info.n_epochs} epochs on this dataset\n"
        f"By default, you'll be charged for ~{cost_info.n_epochs * cost_info.n_billing_tokens} tokens"
    )