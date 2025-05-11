import torch.nn as nn
import torch

class XYZ(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(9, 100)
        self.linear2 = nn.Linear(100, 1)
    

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def code_processor(code: str):
    code = [int(i) for i in code]
    code = torch.tensor(code)
    code = code.to(torch.float32)
    code = code.view(-1, 9)
    return code


def map_code(code):
    code = code_processor(code)
    mapper = XYZ()
    mapper.load_state_dict(torch.load('ViT.pt', weights_only=True))
    mapped_code = mapper(code)
    return mapped_code


