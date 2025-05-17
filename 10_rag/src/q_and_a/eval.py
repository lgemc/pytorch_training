from typing import Callable, Dict, List, Tuple

from q_and_a.forward import ForwardType
from q_and_a.picking.lib import PickerType
from data.q_and_a.eval_with_answers import EvalWithAnswers

def evaluate(
    forward_fn: ForwardType,
    picker_fn: PickerType,
    eval_dataset: EvalWithAnswers,
    log_each: int = 100,
) -> float:
    """
    Evaluate the model on the evaluation dataset.

    Args:
        forward_fn: The forward function to use for generating the response.
        picker_fn: The function to use for picking the best option.
        eval_dataset: The evaluation dataset.
        log_each: The number of samples to log accuracy for.

    Returns:
        The accuracy of the model on the evaluation dataset.
    """
    correct = 0
    total = len(eval_dataset)

    for i in range(total):
        item = eval_dataset[i]
        question = item["question"]
        options = item["options"]
        answer_idx = item["answer_idx"]

        # Get the model's response
        response = forward_fn(question, options)

        # Pick the best option
        picked_idx = picker_fn(response)

        if picked_idx == answer_idx:
            correct += 1

        if i % log_each == 0:
            # print current accuracy
            print(f"Accuracy at {i}: {correct / (i + 1):.2f}")

    return correct / total