import torch
from torch.nn import Module

class OptionsPicker(Module):
    """
    OptionsPicker is a class that provides a way to select options based on the provided input.
    It uses the Llama32 model and tokenizer to generate predictions and select the best option.
    """

    def __init__(self, model, tokenizer, options=None, device="cuda"):
        """
        Initialize the OptionsPicker with a model, tokenizer, and options.
        Options
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.options = options if options is not None else []

    def _get_option_ids(self):
        """
        Convert the options to input IDs using the tokenizer.
        Returns:
            A list of input IDs for each option.
        """
        option_ids = []
        for option in self.options:
            inputs = self.tokenizer(option, return_tensors="pt").to(self.device)
            option_ids.append(inputs["input_ids"][0][1].item())
        return option_ids

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model to generate predictions.
        Args:
            input_ids: Input IDs for the model.
            attention_mask: Attention mask for the model.
        Returns:
            Probabilities for each option.
        """
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
        option_ids = self._get_option_ids()
        option_probs = []
        for option_id in option_ids:
            option_probs.append(probs[option_id].item())
        return torch.tensor(option_probs)


