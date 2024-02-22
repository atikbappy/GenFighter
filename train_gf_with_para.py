#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import itertools
import json
import math
import pickle
import random
import time

import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import CONFIG
from dataset import load_nlp_dataset, prepare_dataset_bert
from helpers import (BaseModel, BaseModelDataset, FineTuneScoringModel,
                     FineTuneT5Paraphraser, ScoringDataset, T5ParaDataSet,
                     flatten_double_list, get_confidences, get_paraphrases,
                     val_accuracy)
from model import BertClassifierDARTS
from pathlib import Path
from rich import box
from rich.console import Console
from rich.table import Column, Table
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AdamW, AutoModelForSeq2SeqLM, AutoTokenizer,
                          RobertaModel, RobertaTokenizer,
                          get_linear_schedule_with_warmup)
from utils import evaluate_batch_single, evaluate_without_attack, get_preds

console = Console(record=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device


# In[ ]:


parser = argparse.ArgumentParser()
# settings

parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument('--epochs', default=10, type=int)

args = parser.parse_args()


# In[2]:


dataset_name = args.dataset_name # 'agnews' | 'imdb' | 'sst'
model_type = CONFIG["model_type"] # 'bert-base-uncased' | 'roberta-base'
EPOCHS = args.epochs

BATCH_SIZE = {
    'sst': 128,
    'agnews': 128,
    'imdb': 64,
}

SETTINGS = {
    'commit': True
}
folder = dataset_name + '_' + model_type

Path(f"./models/{folder}").mkdir(parents=True, exist_ok=True)


# In[4]:


MAX_LENGTH = CONFIG["max_len"][dataset_name]
NUM_OF_PARA = CONFIG["num_of_para"][dataset_name]

train_dataset, val_dataset, test_dataset = load_nlp_dataset(
    dataset_name,
)
original_sentences = train_dataset['text']
labels = train_dataset['label']


# In[6]:


from copy import deepcopy

# learning_rate=1e-8 if training later
def train_base_model(step_number, sentences, labels, learning_rate=3e-5, load_path=None, save_path=None):
    # Start from fresh
    load_path = load_path or (None if step_number == 0 else f'./models/{folder}/{dataset_name}{step_number}.pt')
    save_path = save_path or f'./models/{folder}/{dataset_name}_with_para_{step_number + 1}.pt'
    patience = 2
    epochs = EPOCHS
    grad_clip = 3
    batch_size = BATCH_SIZE[dataset_name]
    
    train_iter, val_iter, test_iter, tokenizer = prepare_dataset_bert(
        model_type, 
        dataset_name, 
        batch_size=batch_size,
        max_len=MAX_LENGTH,
        device=device,
    )

    # Train iter must be replaced
    dataset = BaseModelDataset(
        tokenizer, sentences, MAX_LENGTH, device, labels
    )
    train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print(len(train_iter.dataset))
    print(len(train_iter))
    
    model = BertClassifierDARTS(
        model_type=model_type, 
        freeze_bert=False, 
        output_dim=CONFIG["output_dim"][dataset_name], 
        ensemble=0, 
        device=device
    )
    
    model.init_linear()
    if load_path:
        model.load_state_dict(torch.load(load_path))
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
    # opt = optim.AdamW(model.parameters(), lr=learning_rate)
    opt = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-10)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=len(train_iter) * epochs)
    
    best_val_loss = 9999
    cur_patience = 0
    max_val_accuracy = 0

    for epoch in range(0, epochs):
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

        # if SETTINGS['commit']:  
        #     if v_acc > max_val_accuracy:
        #         max_val_accuracy = v_acc
        #         print("Best val_loss changed. Saving...")
        #         torch.save(model.state_dict(), save_path)

        print("Best val_loss changed. Saving...")
        torch.save(model.state_dict(), save_path)
    


# ### Train classifier with paraphrases

# In[8]:


paraphrases = get_paraphrases(
    CONFIG["paraphraser_name"],  
    original_sentences, 
    NUM_OF_PARA, 
    device, 
    # load_path=f"./models/{folder}/paraphraser{CONFIG['version']}.pt", # Warning: Can be None
    load_path=None,
    max_len=CONFIG["paraphraser_max_len"][dataset_name],
    model_type=CONFIG["paraphraser_type"],
    batch_size=CONFIG["paraphraser_batch_size"][dataset_name],
    disable_tqdm=False,
)

Path(f"./data/paras/{folder}").mkdir(parents=True, exist_ok=True)

with open(f"./data/paras/{folder}/paras_{CONFIG['version']}.pkl", 'wb') as f:
    pickle.dump(paraphrases, f)

# with open(f'./data/paras/{folder}/paras_{CONFIG["version"]}.pkl', 'rb') as f:
#     paraphrases = pickle.load(f)

print("Paraphrases feching done.\n")
train_base_model(
    0, 
    original_sentences + flatten_double_list(paraphrases), 
    labels + flatten_double_list(
        [[l] * NUM_OF_PARA for l in labels]
    ),
    learning_rate=3e-5,
    # load_path = f'./models/{folder}/{dataset_name}_base.pt',
    save_path = f'./models/{folder}/{dataset_name}_{CONFIG["version"]}.pt',
)


# In[9]:


print("All Done")


# In[ ]:




