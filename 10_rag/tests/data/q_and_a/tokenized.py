import unittest

from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B"

from data.q_and_a.train_and_eval import TrainAndEval
from data.q_and_a.eval_with_answers import EvalWithAnswers
from data.q_and_a.prompted import Prompted
from data.q_and_a.tokenized import Tokenized

class TestTokenized(unittest.TestCase):
    def test_tokenized(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        tokenizer.pad_token = tokenizer.eos_token

        eval_dataset = TrainAndEval("../../../data/pubmed_QA_eval.json")
        eval_with_answers = EvalWithAnswers(eval_dataset)
        prompted = Prompted(eval_with_answers, mock_prompter)
        tokenized = Tokenized(tokenizer, prompted)
        tokenized_element = tokenized[0]

        decoded = tokenizer.decode(tokenized_element["input_ids"])
        self.assertIn(prompted[0][0][:10], decoded)

        self.assertIn(tokenized_element["labels"].item(), range(4))

def mock_prompter(question, options, _):
    return f"Question: {question}, Options: {options}"

