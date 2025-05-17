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


class PromptedEhvoy(Dataset):
    def __init__(self, dataset: EhovyRaceDataset, build_prompt=prompt_with_question, include_answer=False):

        self.build_prompt = build_prompt
        self.include_answer = include_answer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.build_prompt(self.dataset[idx], include_answer=self.include_answer), self.dataset[idx]['answer']


