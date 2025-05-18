from typing import List

def prompt(
    question: str,
    options: List[str],
    augmented_items: List[str],
) -> str:
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

Context:
{context}

Question: {question}

Options:
{options_str}

Answer(pick one between A, B, C and D):"""