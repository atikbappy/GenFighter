#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse

import numpy as np 
import pandas as pd 

import random
import tensorflow as tf
import torch

from config import CONFIG
from dataset import prepare_dataset_bert
from helpers import val_accuracy
from model import BertClassifierDARTS
from pathlib import Path
from utils import evaluate_without_attack, get_preds, evaluate_batch_single

from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm


# In[2]:


parser = argparse.ArgumentParser()
# settings

parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument('--epochs', default=10, type=int)

args = parser.parse_args()


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


epochs = args.epochs
batch_size = {
    'sst': 128,
    'agnews': 128,
    'imdb': 64,
}
grad_clip = 3

dataset_name = args.dataset_name # 'agnews' | 'agnews' | 'imdb'
MAX_LEN = CONFIG["max_len"][dataset_name]
model_type = CONFIG['model_type'] # 'bert-base-uncased'
folder = dataset_name + '_' + model_type
save_path = f'./models/{folder}/{dataset_name}_base.pt'
patience = 2

train_iter, val_iter, test_iter, tokenizer = prepare_dataset_bert(
    model_type, 
    dataset_name, 
    batch_size=batch_size[dataset_name],
    max_len=MAX_LEN,
    device=device,
)

print("Train:", len(train_iter.dataset))
print("Val:", len(val_iter.dataset))
print("Test:", len(test_iter.dataset))

Path(f"./models/{folder}").mkdir(parents=True, exist_ok=True)


# In[4]:


model = BertClassifierDARTS(
    model_type=model_type, 
    freeze_bert=False, 
    output_dim=CONFIG["output_dim"][dataset_name], 
    ensemble=0, 
    device=device
)
model.init_linear()
model = model.to(device)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
opt = optim.AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=len(train_iter)*epochs)


# In[5]:


model.train()
loss_func = nn.CrossEntropyLoss()

best_val_loss = 9999
cur_patience = 0
max_val_accuracy = 0


# In[6]:


for epoch in range(0, epochs):
    print(f"Epoch: {epoch + 1}")
    
    total_train = 0
    model.train() 

    for batch in tqdm(train_iter):
        preds, loss, acc = evaluate_batch_single(model, batch, allow_grad=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        scheduler.step()

    model.inference = True
    val_loss, preds = evaluate_without_attack(model, val_iter)
    print("Val Loss: " + str(val_loss))
    v_acc = val_accuracy(val_iter, preds)
    print("Val Acc: " + str(v_acc))
    model.inference = False

    # if v_acc > max_val_accuracy:
    #     max_val_accuracy = v_acc
    #     print("Best val_loss changed. Saving...")
    #     torch.save(model.state_dict(), save_path)

    
    print("Best val_loss changed. Saving...")
    torch.save(model.state_dict(), save_path)

    # if best_val_loss > val_loss:
    #     cur_patience = 0
    #     best_val_loss = val_loss
    #     if save_path != "":
    #         print("Best val_loss changed. Saving...")
    #         torch.save(model.state_dict(), save_path)
    # else:
    #     cur_patience += 1
    #     if cur_patience >= patience:
    #         break


# In[ ]:




