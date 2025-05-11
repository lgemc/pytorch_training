import torch

import torch.nn as nn
from model.lora import LoraLinear
from transformers import AutoModelForCausalLM, AutoTokenizer

class Llama32Lora(nn.Module):
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
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
        for param in model.parameters():
            param.requires_grad = False

        for layer in model.model.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn.q_proj = LoraLinear(layer.self_attn.q_proj, r=16)
                layer.self_attn.k_proj = LoraLinear(layer.self_attn.k_proj, r=16)
                layer.self_attn.v_proj = LoraLinear(layer.self_attn.v_proj, r=16)
                layer.self_attn.o_proj = LoraLinear(layer.self_attn.o_proj, r=16)

        return model

    def print_parameters(self):
        params_without_lora = 0
        params_with_lora = 0
        for name, param in self.model.named_parameters():
            if 'self_attn' in name and 'linear_layer' in name:
                params_without_lora += param.numel()
            if param.requires_grad:
                params_with_lora += param.numel()

        print(
            f'Parámetros sin LoRA: {params_without_lora:,} || Parámetros con LoRA: {params_with_lora:,} || Porcentaje de parámetros con LoRA: {100 * params_with_lora / (params_without_lora + params_with_lora):.2f}%')

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

