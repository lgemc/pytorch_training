import torch.nn as nn

from transformers import AutoTokenizer

class Llama32Tokenizer(nn.Module):
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = self.load_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def forward(self,
                text: str,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
                ):
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        ).to(self.device)
        return inputs

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)