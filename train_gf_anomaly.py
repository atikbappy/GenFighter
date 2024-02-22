#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse

import numpy as np 
import pandas as pd 

import joblib
import pickle
import torch

import sys

from config import CONFIG
from dataset import load_nlp_dataset, prepare_dataset_bert, prepare_single_bert
from helpers import get_paraphrases, val_accuracy
from model import BertClassifierDARTS
from pathlib import Path
from utils import evaluate_without_attack, get_preds, evaluate_batch_single

from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

from tqdm import tqdm


# In[2]:


parser = argparse.ArgumentParser()
# settings

parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument('--anomaly_model', default='gmm', type=str)

args = parser.parse_args()


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


batch_size = 128
ANOMALY_ALGO = args.anomaly_model # 'gmm' | 'if'
dataset_name = args.dataset_name # 'imdb' | 'agnews' | 'sst'
model_type = CONFIG["model_type"] # 'bert-base-uncased' | 'roberta-base'
folder = dataset_name + '_' + model_type
load_path = f'./models/{folder}/{dataset_name}_{CONFIG["version"]}.pt'

MAX_LEN = CONFIG["max_len"][dataset_name]
USE_PARAPHRASES = CONFIG["use_paraphrases_to_train_anomlay"][dataset_name]

train_dataset, val_dataset, test_dataset = load_nlp_dataset(
    dataset_name
)


# In[4]:


model = BertClassifierDARTS(
    model_type=model_type, 
    freeze_bert=False, 
    output_dim=CONFIG["output_dim"][dataset_name], 
    ensemble=0, 
    device=device
)
model.load_state_dict(torch.load(load_path))
model = model.to(device)


# In[5]:


NUM_OF_PARA = CONFIG["num_of_para"][dataset_name]

original_sentences = train_dataset['text']
labels = train_dataset['label']
texts = list()

if USE_PARAPHRASES:
    # Path(f"./models/{folder}").mkdir(parents=True, exist_ok=True)
    
    # paraphrases = get_paraphrases(
    #     CONFIG["paraphraser_name"],  
    #     original_sentences, 
    #     NUM_OF_PARA, 
    #     device, 
    #     # load_path=f"./models/{folder}/paraphraser_{CONFIG['version']}.pt", # Warning: Can be None
    #     max_len=CONFIG["paraphraser_max_len"][dataset_name],
    #     model_type=CONFIG["paraphraser_type"],
    #     batch_size=CONFIG["paraphraser_batch_size"][dataset_name],
    #     disable_tqdm=False,
    # )
    with open(f"./data/paras/{folder}/paras_{CONFIG['version']}.pkl", 'rb') as f:
        paraphrases = pickle.load(f)
    for s, p in zip(original_sentences, paraphrases):
        texts.append(s)
        texts.extend(p)
else:
    texts = original_sentences

data_iter = prepare_single_bert(
    texts, 
    tokenizer=model.tokenizer, 
    batch_size=batch_size, 
    max_len=MAX_LEN,
    device=device,
    shuffle=True,
)
X = np.empty((len(data_iter.dataset), 768))


# In[6]:


model.eval() 
index = 0
with torch.set_grad_enabled(False):
    for batch in tqdm(data_iter):
        seq = batch['input_ids']
        attn_masks = batch['attention_mask']
        cont_reps = model.bert_layer(input_ids=seq, attention_mask=attn_masks)
        lhs = cont_reps.last_hidden_state[:, 0]
        for logits in lhs:
            X[index] = logits.cpu().numpy()
            index += 1


# ## Finding out BIC

# In[7]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

bic_scores = list()
for n_components in range(2, 13):
    anomaly_model = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    bic_scores.append((anomaly_model.bic(X), n_components))
    print(bic_scores[-1])

print(bic_scores)
print(np.argmin([b for b, _ in bic_scores]))

min_index = np.argmin([b for b, _ in bic_scores])
n_components = bic_scores[min_index][1]
print(n_components)


# ## Training the Anomaly model

# In[ ]:

# n_components = 7
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if ANOMALY_ALGO == 'if':
    anomaly_model = IsolationForest(random_state=0).fit(X)
elif ANOMALY_ALGO == 'gmm':
    anomaly_model = GaussianMixture(n_components=n_components, random_state=0).fit(X)


# In[11]:


Path(f"./models/anomaly").mkdir(parents=True, exist_ok=True)
joblib.dump(anomaly_model, f'./models/anomaly/{dataset_name}_{model_type}_{ANOMALY_ALGO}_{CONFIG["version"]}.pkl')


# ## Finding out threshold

# In[13]:


anomaly_model = joblib.load(f'./models/anomaly/{dataset_name}_{model_type}_{ANOMALY_ALGO}_{CONFIG["version"]}.pkl')
anomaly_scores = anomaly_model.score_samples(X)


# **Min, Max, Avg**

# In[14]:


print(anomaly_scores.min(), anomaly_scores.max(), anomaly_scores.mean())


# In[15]:


import matplotlib.pyplot as plt

# Plot histogram of anomaly scores
plt.hist(anomaly_scores, bins=50, density=True, alpha=0.5)
plt.title('Anomaly Score Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.show()


# **5 percentile, 10 percentile and 95 percentile**

# In[ ]:


print("5 percentile, 10 percentile and 95 percentile")


# In[16]:


print(np.percentile(anomaly_scores, 5), np.percentile(anomaly_scores, 10), np.percentile(anomaly_scores, 95))


# In[ ]:





# In[ ]:




