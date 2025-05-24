from torch.utils.data import Dataset

from data.q_and_a.eval_with_answers import EvalWithAnswers
from q_and_a.prompts import Prompter
from datasets import Dataset as HFDataset

OPTIONS =  ['A', 'B', 'C', 'D']
class Prompted(Dataset):
    def __init__(
            self,
            dataset: EvalWithAnswers,
            prompter: Prompter,
            options: list = OPTIONS,
    ):
        self.dataset = dataset
        self.prompter = prompter
        self.options = options

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        item = self.dataset[item]
        answer = item["answer_idx"]
        options = item["options"]
        question = item["question"]

        return self.prompter(question, options, [item["excerpt"]], index_to_answer(item["answer_idx"], self.options)), answer



def index_to_answer(index: int, options: list) -> str:
    """
    Convert an index to an answer string.
    """
    return options[index]


def to_transformers_dataset(dataset: Prompted) -> HFDataset:
    data = [{"text": dataset[i][0], "label": index_to_answer(dataset[i][1], OPTIONS)} for i in range(len(dataset))]
    hf_dataset = HFDataset.from_list(data)

    return hf_dataset