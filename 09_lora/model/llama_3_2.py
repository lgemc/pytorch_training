import torch.nn
import torch
from transformers import AutoModelForCausalLM


class Llama32(torch.nn.Module):
    def __init__(
            self,
            model_name: str,
            do_sample: bool = True,
            device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.do_sample = do_sample
        self.model = self.load_model()

    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float32).to(self.device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def generate(self, input_ids, attention_mask=None, max_length=50):
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=self.do_sample,
            pad_token_id=self.model.config.pad_token_id,
            max_length=max_length)
        return generated_ids