from typing import List, Callable, Optional

# Prompter receive a question, a list of possible answers and an augmented list of information
# that can serve as context for the question, and returns a prompted version of the question
Prompter = Callable[[str, List[str], Optional[List[str]]], str]

def prompt(
    question: str,
    options: List[str],
    augmented_items: List[str] = None,
) -> str:
    context = ""
    if augmented_items is not None:
        context = "\n".join(augmented_items)

    options_str = "\n".join(
        [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
    )

    """
    Generates a prompt for the language model based on the question, options, and augmented items.

    Args:
        question (str): The question to ask.
        options (List[str]): The list of options to choose from.
        augmented_items (List[str]): The augmented items to include in the prompt.

    Returns:
        str: The formatted prompt string.
    """
    return f"""You are an expert in multiple-choice questions. Your task is to select the best answer from the given options based on the provided context.
{context}
Question: {question}

Options:
{options_str}

Between A, B, C and D the best option is """