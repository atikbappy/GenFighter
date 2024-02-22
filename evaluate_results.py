#!/usr/bin/env python
# coding: utf-8

# In[16]:


import argparse
import json

import numpy as np
import pandas as pd
import torch
from dataset import load_nlp_dataset
from datasets.utils.logging import set_verbosity_error
from sklearn.metrics import accuracy_score, f1_score
from utils import cal_true_success_rate, dataset_mapping, load_attacker

set_verbosity_error()

import random
from contextlib import contextmanager

import joblib
import numpy as np
import OpenAttack as oa
import tensorflow as tf
import torch
from config import CONFIG, ATTACK_CONFIG
from dataset import load_nlp_dataset, prepare_single_bert
from datasets import Dataset
from helpers import (BaseModel, BaseModelDataset, FineTuneScoringModel,
                     ScoringDataset, T5ParaDataSet, dump_adv_examples,
                     flatten_double_list, get_confidences, get_paraphrases,
                     val_accuracy)
from model import BertClassifierDARTS
from results import Result
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import (cal_true_success_rate, dataset_mapping, get_preds,
                   load_attacker)

random.seed(12)
torch.manual_seed(12)
tf.random.set_seed(12)
np.random.seed(12)


# In[ ]:


parser = argparse.ArgumentParser()
# settings

parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument('--attacker', required=True, type=str)

args = parser.parse_args()


# In[17]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Settings

# In[3]:


dataset_name = args.dataset_name # 'agnews' | 'sst' | 'imdb'
model_type = CONFIG["model_type"] # 'bert-base-uncased' | 'roberta-base'
folder = dataset_name + '_' + model_type
attacker_name = args.attacker # 'TextFooler' | 'PWWS' | 'BertAttack'

rng = np.random.default_rng(ATTACK_CONFIG["test_seed"])


# File Paths

# In[10]:


alt_adv_path = f'./data/attack_results/{dataset_name}/{model_type}_{attacker_name}_{ATTACK_CONFIG["adv_file_post_fix"]}_{CONFIG["version"]}.json'


# Load Data from Files

# In[11]:


with open(alt_adv_path, 'r') as f:
    alt_adv = json.load(f)


# ### Our Model

# In[12]:


our_result = Result(alt_adv)
our_result.print_stats()


# In[ ]:





# In[ ]:




