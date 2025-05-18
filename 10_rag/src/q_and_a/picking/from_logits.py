from typing import Callable

import torch

def _from_logits(
        tokenizer,
        logits: torch.Tensor,
        options: list[str],
) -> int:
    """
    Picks an option from a list of options based on the given logits.

    Args:
        tokenizer: The tokenizer to use for encoding the options.
        logits (list[float]): The logits for each option.
        options (list[str]): The list of options to choose from.

    Returns:
        int: The index of the chosen option.
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=0).tolist()

    options_representation = [
        tokenizer.encode(option, add_special_tokens=False)[0]
        for option in options
    ]


    # Calculate the score for each option
    scores = []
    for _, option in enumerate(options_representation):
        score = probs[option]
        scores.append(score)

    # Choose the option with the highest score
    chosen_option = scores.index(max(scores))
    return chosen_option

def build_from_logits(
    tokenizer,
    options: list[str],
) -> Callable[[torch.Tensor], int]:
    """
    Builds a function that can be used to pick an option from a list of options based on the given logits.

    Args:
        tokenizer: The tokenizer to use for encoding the options.
        options (list[str]): The list of options to choose from.

    Returns:
        Callable[[torch.Tensor], int]: A function that takes logits and returns the index of the chosen option.
    """
    return lambda model_out: _from_logits(tokenizer, model_out, options)