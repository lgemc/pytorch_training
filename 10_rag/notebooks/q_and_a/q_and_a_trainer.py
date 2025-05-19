#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data.q_and_a.train_and_eval import TrainAndEval
from data.q_and_a.eval_with_answers import EvalWithAnswers
from q_and_a.prompts import prompt
from data.q_and_a.prompted import Prompted
import torch.optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


# In[12]:


from huggingface_hub import login
login("{token}")


# # First, load the data
# 
# We are going to load the data used for train or modify our classification task.

# In[3]:


class Tokenized(Dataset):
    def __init__(self, tokenizer, dataset: Prompted, max_length=2000):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        text, answer = self.dataset[idx]

        result = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",)
        labels = torch.tensor(answer, dtype=torch.long)

        return {
            "input_ids": result["input_ids"].squeeze(0),
            "attention_mask": result["attention_mask"].squeeze(0),
            "labels": labels,
        }


# In[4]:


MODEL_NAME = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token

train_dataset = TrainAndEval("../../data/pubmed_QA_train.json")
test_dataset = TrainAndEval("../../data/pubmed_QA_eval.json")
train_with_answers = EvalWithAnswers(train_dataset)
test_with_answers = EvalWithAnswers(test_dataset)
train_prompted= Prompted(train_with_answers, prompt)
test_prompted = Prompted(test_with_answers, prompt)
train_tokenized = Tokenized(tokenizer, train_prompted)
test_tokenized = Tokenized(tokenizer, test_prompted)


# In[5]:


len(train_tokenized), len(test_tokenized)


# In[6]:


# per now use a subset
from torch.utils.data import Subset

train_tokenized = Subset(train_tokenized, range(0, 2000))
test_tokenized = Subset(test_tokenized, range(0, 200))


# In[7]:


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    load_in_8bit=True,
    pad_token_id=tokenizer.pad_token_id,
)
model


# In[8]:


for name, param in model.named_parameters():
    if "score" not in name:
        print(f"grad non required on:{name}")
        param.requires_grad = False
    else:
        print(f"requires grad: {name}")


# In[9]:


from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,  # Rank of LoRA matrices (lower = less memory)
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Depends on model architecture
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# In[10]:


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
)

trainer.train()


# In[ ]:


print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0))
print("Supports FP16:", torch.cuda.get_device_capability(0))


# In[ ]:


trainer.save_model("./last-checkpoint/trainer")
tokenizer.save_pretrained("./last-checkpoint/tokenizer")


# In[ ]:


model.save_pretrained("./last-checkpoint/model")


# In[ ]:


import torch
from pathlib import Path

from transformers import AutoModelForSequenceClassification,AutoTokenizer

from data.q_and_a.train_and_eval import TrainAndEval
from data.q_and_a.eval_with_answers import EvalWithAnswers

from models_.building.llama_tokenizer import  load_tokenizer

from data.pubmed.from_json import FromJsonDataset
from data.pubmed.contents import ContentsDataset

from storage.faiss_ import FaissStorage

from rag.tokenization.llama import build_tokenizer_function
from rag.quering import build_querier
import os
from q_and_a.forward import build_enhanced_forwarder
from q_and_a.prompts import prompt
from q_and_a.picking.from_logits import build_from_logits
from q_and_a.eval import evaluate
from q_and_a.forward import build_forwarder

train = TrainAndEval("../../data/pubmed_QA_train.json")
evaluationData = TrainAndEval("../../data/pubmed_QA_eval.json")
evaluateWithAnswers = EvalWithAnswers(evaluationData)

augmented_data = FromJsonDataset(json_file="../../data/pubmed_500K.json")
augmented_data = ContentsDataset(augmented_data)

from huggingface_hub import notebook_login
notebook_login()

storage = FaissStorage(
    dimension=800,
)

storage.load("../../outputs/store/pubmed_500K.index")


# In[ ]:


tokenizer_rag = load_tokenizer()
tokenizer_fn = build_tokenizer_function(tokenizer_rag)

querier = build_querier(storage, augmented_data, tokenizer_fn)
storage = FaissStorage(
    dimension=800,
)


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[ ]:


model.eval()
forward = build_forwarder(
    model,
    tokenizer,
    querier,
    k_augmentations=1,
    prompt_builder=prompt,
    device=device,
)

forward_and_get_arg_max = lambda question, options: forward(
    question,
    options=options,
)

def pick_from_classifier(out):
    return torch.argmax(out.logits[0])

accuracy = evaluate(
    forward_fn=forward_and_get_arg_max,
    picker_fn=pick_from_classifier,
    eval_dataset=evaluateWithAnswers,
)

print(f"Accuracy: {accuracy:.2f}")


# In[ ]:




