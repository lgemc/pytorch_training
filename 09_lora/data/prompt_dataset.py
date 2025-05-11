from torch.utils.data import Dataset

from data.ehovy_race import EhovyRaceDataset

def prompt_with_question(example: dict, include_answer=False) -> str:
    options_str = "\n".join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(example['options'])])
    answer = f" {example['answer']}" if include_answer else ""
    prompt = f"Context: {example['article']}\n\n" + \
        f"Question: {example['question']}\n\n" + \
        f"Options:\n{options_str}\n\n" + \
        f"Answer:{answer}" # The model is expected to fill this part

    return prompt

def simple_prompt_with_question(example: dict) -> str:
    """
    A simpler version of the prompt without the "Context" and "Question" labels.
    """
    options_str = "\n".join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(example['options'])])
    prompt = f"{example['article']}\n\n" + \
        f"{example['question']}\n\n" + \
        f"{options_str}\n\n" + \
        f"Answer:"  # The model is expected to fill this part

    return prompt

def extract_answer(output: str) -> str:
    """
    Extract the answer from the model output.
    The answer is expected to be in the format "Answer: A", "Answer: B", etc.
    """
    if "Answer:" in output:
        answer_part = output.split("Answer:")[-1].strip()
        if answer_part and len(answer_part) == 1 and answer_part.isalpha():
            return answer_part.upper()
    return None


class PromptedEhvoy(Dataset):
    def __init__(self, dataset: EhovyRaceDataset, build_prompt=prompt_with_question, include_answer=False):

        self.build_prompt = build_prompt
        self.include_answer = include_answer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.build_prompt(self.dataset[idx], include_answer=self.include_answer), self.dataset[idx]['answer']


