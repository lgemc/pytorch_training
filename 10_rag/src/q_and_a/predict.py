from typing import List, Tuple

from q_and_a.forward import ForwardType
from q_and_a.picking.lib import PickerType
from data.q_and_a.eval_with_answers import EvalWithAnswers

def predict(
    forward_fn: ForwardType,
    picker_fn: PickerType,
    eval_dataset: EvalWithAnswers,
    log_each: int = 100,
) -> List[Tuple[int, int]]:
    """
    Evaluate the model on the evaluation dataset.

    Args:
        forward_fn: The forward function to use for generating the response.
        picker_fn: The function to use for picking the best option.
        eval_dataset: The evaluation dataset.
        log_each: The number of samples to log accuracy for.

    Returns:
        The id and the picked option of the model on the evaluation dataset.
    """
    total = len(eval_dataset)
    responses = []
    for i in range(total):
        item = eval_dataset[i]
        question = item["question"]
        options = item["option"]

        # Get the model's response
        response = forward_fn(question, options)

        # Pick the best option
        picked_idx = picker_fn(response)

        responses.append((i, picked_idx))

        if i != 0 and i% log_each == 0:
            print(f"Processed {i/total}%")

    return responses