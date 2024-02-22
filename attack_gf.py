#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets.utils.logging import set_verbosity_error

set_verbosity_error()

import argparse
import sys

import random
from contextlib import contextmanager

import joblib
import numpy as np
import OpenAttack as oa
import tensorflow_text
import tensorflow as tf
import torch
from attack_utils import AnomalyClassifier, pick_less_anomalous_text
from config import CONFIG, ATTACK_CONFIG
from dataset import load_nlp_dataset, prepare_single_bert
from datasets import Dataset
from helpers import (BaseModel, BaseModelDataset, FineTuneScoringModel,
                     ScoringDataset, T5ParaDataSet, dump_adv_examples,
                     flatten_double_list, get_anomalous_scores, get_confidences,
                     get_paraphrases, val_accuracy)
from model import BertClassifierDARTS
from pathlib import Path
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

torch.cuda.manual_seed(12)


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


dataset_name = args.dataset_name # 'agnews' | 'imdb' | 'sst'
model_type = CONFIG['model_type'] # 'bert-base-uncased' | 'roberta-base'
folder = dataset_name + '_' + model_type
load_path = f'./models/{dataset_name}_{model_type}/{dataset_name}_{CONFIG["version"]}.pt'

MAX_LEN = CONFIG['max_len'][dataset_name]
NUM_OF_PARA = CONFIG["num_of_para"][dataset_name]

inference_temp = 0.01
ANOMALY_ALGO = args.anomaly_model # 'gmm' | 'if'
clf = joblib.load(f'models/anomaly/{dataset_name}_{model_type}_{ANOMALY_ALGO}_{CONFIG["version"]}.pkl')

rng = np.random.default_rng(ATTACK_CONFIG["test_seed"])


# In[4]:


model = BertClassifierDARTS(
    model_type=model_type, 
    freeze_bert=True, # was False
    output_dim=CONFIG["output_dim"][dataset_name], 
    ensemble=0, 
    device=device
)
model.load_state_dict(torch.load(load_path))
model = model.to(device)
model.eval()


# In[5]:


print(CONFIG)


# In[6]:


print(ATTACK_CONFIG)


# In[7]:


Path(f'./data/attack_results/{dataset_name}').mkdir(parents=True, exist_ok=True)
NUMBER_TO_ATTACK = 1000

_, _, test_dataset = load_nlp_dataset(dataset_name)
test_dataset = test_dataset.select(
    rng.choice(len(test_dataset), NUMBER_TO_ATTACK, replace=False)[ATTACK_CONFIG["test_split_starts"]: ATTACK_CONFIG["test_split_ends"]]
)
test_dataset = test_dataset.map(dataset_mapping)

tokenizer = AutoTokenizer.from_pretrained(model_type)
victim = AnomalyClassifier(
    model, 
    tokenizer, 
    dataset_name,
    model_type,
    clf,
    MAX_LEN,
    batch_size=1, 
    device=device,
)

for attacker_name in ATTACK_CONFIG["attacker_names"]:
    print("\n" + attacker_name + " attack starts: \n")
    attacker = load_attacker(attacker_name)
    attack_eval = oa.AttackEval(
        attacker, 
        victim,
        metrics=[oa.metric.Modification(None)],
    )
    
    adversarials, result = attack_eval.eval(
        test_dataset, 
        visualize=ATTACK_CONFIG["visualize_attack"],
    )
    if ATTACK_CONFIG["save_adv_examples"]:
        dump_adv_examples(
            f'./data/attack_results/{dataset_name}/{model_type}_{attacker_name}_{ATTACK_CONFIG["adv_file_post_fix"]}_{CONFIG["version"]}.json', 
            adversarials
        )

    our_result = Result(adversarials)
    our_result.print_stats()

    print("\n" + attacker_name + " attack ends. \n")
    


# ### Compute Clean Accuracy

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_type)
victim = AnomalyClassifier(
    model, 
    tokenizer, 
    dataset_name, 
    model_type,
    clf,
    MAX_LEN,
    batch_size=1,  
    device=device
)
_, _, test_dataset = load_nlp_dataset(dataset_name)
test_dataset = test_dataset.map(dataset_mapping)


# In[ ]:


preds = victim.get_pred(test_dataset['text'])
from sklearn.metrics import accuracy_score
print(accuracy_score(test_dataset['label'], preds))


# 0.9231191652937946

# In[ ]:





# In[ ]:




