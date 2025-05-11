import unittest
from textgradlib.engine import Engine
from model.llama_3_2 import Llama32
from model.llama_3_2_tokenizer import Llama32Tokenizer

model_1B = "meta-llama/Llama-3.2-1B"

class TestTextGradEngine(unittest.TestCase):
    def test_textgrad_engine(self):
        model = Llama32(model_1B, "cuda")
        tokenizer = Llama32Tokenizer(model_1B)

        engine = Engine(model, tokenizer, max_length=500)

