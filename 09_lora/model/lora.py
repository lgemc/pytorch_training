import torch
import math

class LoraLinear(torch.nn.Module):
    def __init__(self, linear_layer, alpha = 1, r = 1, device = "cuda"):
        super().__init__()
        self.linear_layer = linear_layer.to(torch.float32) # Se cambia el tipo de la capa a float32 para evitar errores durante el entrenamiento
        self.r = r
        fan_in = self.linear_layer.in_features
        fan_out = self.linear_layer.out_features
        self.lora_A = torch.nn.Parameter(torch.zeros((fan_in, r), device=device))
        self.lora_B = torch.nn.Parameter(torch.zeros((r, fan_out), device=device))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.linear_layer.weight.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        if not mode:
            self.merged_weight = (self.linear_layer.weight.transpose(0,1) + self.lora_A @ self.lora_B).to(torch.float16) # Se cambia el tipo de la matriz a float16

    def forward(self, x):
        if self.training:
            x = x.to(torch.float32)
            output = self.linear_layer(x)
            output += x @ self.lora_A @ self.lora_B
            output = output.to(torch.float16)
        else:
            output = x @ self.merged_weight
        return output