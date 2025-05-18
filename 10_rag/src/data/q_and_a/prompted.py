from torch.utils.data import Dataset

from data.q_and_a.eval_with_answers import EvalWithAnswers
from q_and_a.prompts import Prompter

class Prompted(Dataset):
    def __init__(
            self,
            dataset: EvalWithAnswers,
            prompter: Prompter,
    ):
        self.dataset = dataset
        self.prompter = prompter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        item = self.dataset[item]
        answer = item["answer_idx"]
        options = item["options"]
        question = item["question"]

        return self.prompter(question, options, None), answer
