from typing import Any

import torch

from typing import Callable, List, Tuple
import torch.nn.functional as F

ForwardType = Callable[[str, List[str]], Any]


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
    # Generate the prompt
    prompt = prompt_builder(question, options, items)

    result = tokenizer(prompt, return_tensors="pt").to(device)

    # Use the language model to generate a response
    with torch.no_grad():
        return llm(**result, do_sample=False)


def build_forwarder(
    llm,
    tokenizer,
    augmenter: Callable[[str, int], Tuple[List[int], List[str]]],
    k_augmentations: int,
    prompt_builder: Callable[[str, List[str], List[str]], str],
    device: str,
) -> Callable[[str, List[str]], str]:
    """
    Builds a forward function that can be used to generate responses from the language model.

    Returns:
        Callable: A function that takes a question and a list of options and returns the generated response.
    """
    def forward_fn(question: str, options: List[str]) -> Any:
        return forward(
            llm=llm,
            tokenizer=tokenizer,
            augmenter=augmenter,
            k_augmentations=k_augmentations,
            prompt_builder=prompt_builder,
            question=question,
            options=options,
            device=device,
        )

    return forward_fn


def enhanced_forward(
        llm,
        tokenizer,
        augmenter: Callable[[str, int], Tuple[List[int], List[str]]],
        k_augmentations: int,
        prompt_builder: Callable[[str, List[str], List[str]], str],
        question: str,
        options: List[str],
        device: str,
        num_iterations: int = 3,
):
    """
    Performs multiple forward passes, appending the most probable token each time,
    and returns average probabilities across all tokens.

    Args:
        llm: The language model to use for generating the response.
        tokenizer: The tokenizer associated with the language model.
        augmenter (Callable): A function that takes a query string and returns the first k_augmentations in a tuple of
            distances and items.
        k_augmentations (int): The number of augmentations to generate.
        prompt_builder (Callable): A function that builds the prompt for the language model.
            It takes the augmented information, question and options, and returns a formatted string.
        question (str): The question to ask.
        options (List[str]): The list of options to choose from.
        device (str): The device to run the model on ('cpu' or 'cuda').
        num_iterations (int): Number of forward passes to perform (default: 3).

    Returns:
        Tuple containing:
            - List of generated tokens
            - Average probabilities tensor across all tokens
    """
    # Get augmented items
    _, items = augmenter(question, k_augmentations)

    # Generate the initial prompt
    prompt = prompt_builder(question, options, items)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    all_probs = []
    generated_tokens = []

    for _ in range(num_iterations):
        # Forward pass
        with torch.no_grad():
            outputs = llm(input_ids=input_ids)
            logits = outputs.logits

            # Get the last token's logits
            last_token_logits = logits[0][-1]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

            # Get the most probable token
            next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())

            # Append the token to input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Average the probabilities across all iterations
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)

    return generated_tokens, avg_probs

def build_enhanced_forwarder(
    llm,
    tokenizer,
    augmenter: Callable[[str, int], Tuple[List[int], List[str]]],
    k_augmentations: int,
    prompt_builder: Callable[[str, List[str], List[str]], str],
    num_iterations: int,
    device: str,
) -> Callable[[str, List[str]], Tuple[List[int], torch.Tensor]]:
    """
    Builds an enhanced forward function that can be used to generate responses from the language model.

    Returns:
        Callable: A function that takes a question and a list of options and returns the generated response.
    """
    def forward_fn(question: str, options: List[str]) -> Tuple[List[int], torch.Tensor]:
        return enhanced_forward(
            llm=llm,
            tokenizer=tokenizer,
            augmenter=augmenter,
            k_augmentations=k_augmentations,
            prompt_builder=prompt_builder,
            question=question,
            options=options,
            num_iterations=num_iterations,
            device=device,
        )

    return forward_fn