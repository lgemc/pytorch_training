import random

from torch.utils.data import Dataset

from data.q_and_a.train_and_eval import TrainAndEval

class EvalWithAnswers(Dataset):
    """
    A dataset that takes a TrainAndEval dataset and adds the statement to the distractors
    to create a multiple choice question. The statement is inserted at a random position
    in the distractors.

    Returns two item keys: options and answer_idx.
    """
    def __init__(self, dataset: TrainAndEval):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]

        options = item["distractors"]
        # insert the statement at any random position in the options
        index = random.randint(0, len(options))
        options.insert(index, item["statement"])

        item["options"] = options
        item["answer_idx"] = index

        return item
