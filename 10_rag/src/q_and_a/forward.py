from typing import Callable, List, Tuple

import torch


def forward(
        llm,
        tokenizer,
        augmenter: Callable[[str, int], Tuple[List[int], List[str]]],
        k_augmentations: int,
        prompt_builder: Callable[[str, List[str], List[str]], str],
        question: str,
        options: List[str],
        device: str,
):
    """
    Picks an option from a list of options based on the given question.

    Args:
        llm: The language model to use for generating the response.
        augmenter (Callable): A function that takes a query string and returns the first k_augmentations in a tuple of
            distances and items.
        k_augmentations (int): The number of augmentations to generate.
        prompt_builder (Callable): A function that builds the prompt for the language model.
            It takes the augmented information, question and options, and returns a formatted string.
        question (str): The question to ask.
        options (List[str]): The list of options to choose from.

    Returns:
        the response from the language model.
    """
    _, items = augmenter(question, k_augmentations)
    print(f"Augmented items: {items}")
    # Generate the prompt
    prompt = prompt_builder(question, options, items)

    print(f"Prompt: {prompt}")

    result = tokenizer(prompt, return_tensors="pt").to(device)

    # Use the language model to generate a response
    with torch.no_grad():
        return llm(**result, do_sample=False)