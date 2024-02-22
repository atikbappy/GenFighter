#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

import argparse
import OpenAttack as oa
from config import CONFIG, ATTACK_CONFIG
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import *
from dataset import *
from helpers import dump_adv_examples
from pathlib import Path
from utils import *
import random
import numpy
import torch
import tensorflow as tf
import json

from results import Result

random.seed(12)
torch.manual_seed(12)
tf.random.set_seed(12)
np.random.seed(12)


# In[ ]:


parser = argparse.ArgumentParser()
# settings

parser.add_argument("--dataset_name", type=str, required=True)

args = parser.parse_args()


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


dataset_name = args.dataset_name # 'sst' | 'agnews' | 'imdb'
model_type = CONFIG['model_type'] # 'bert-base-uncased' | 'roberta-base'
folder = dataset_name + '_' + model_type
load_path = f'./models/{folder}/{dataset_name}_base.pt'

max_len = CONFIG["max_len"][dataset_name]
attacker_names = ['PWWS', 'TextFooler', 'BertAttack']
inference_temp = 0.01


# In[4]:


model = BertClassifierDARTS(
    model_type=model_type, 
    freeze_bert=True,
    is_training=False,
    inference=True,
    output_dim=CONFIG["output_dim"][dataset_name], 
    ensemble=0, # was: 1 
    device=device
)
if load_path:
    model.load_state_dict(torch.load(load_path))
model = model.to(device)


# In[5]:


model.eval()
_, _, test_iter, _ = prepare_dataset_bert(
    model_type, 
    dataset_name, 
    batch_size=32,
    max_len=max_len,
    device=device
)
preds = get_preds(model, test_iter)
preds = np.argmax(preds, axis=1)
labels = [a['label'] for a in test_iter.dataset]
# f1 = f1_score(labels, preds)
acc = accuracy_score(labels, preds)
print("ACC:", acc)
# print("F1:", f1)


# In[6]:


NUMBER_TO_ATTACK = 1000

rng = np.random.default_rng(ATTACK_CONFIG["test_seed"])

_, _, test_dataset = load_nlp_dataset(dataset_name)
test_dataset = test_dataset.select(rng.choice(len(test_dataset), NUMBER_TO_ATTACK, replace=False))
test_dataset = test_dataset.map(dataset_mapping)

tokenizer = AutoTokenizer.from_pretrained(model_type)
victim = MyClassifier(
    model, 
    tokenizer, 
    batch_size=1, 
    max_len=max_len,
    device=device
)

for attacker_name in attacker_names:
    print("Attack: " + attacker_name + " starts")
    
    attacker = load_attacker(attacker_name)
    attack_eval = oa.AttackEval(
        attacker, 
        victim,
        metrics=[oa.metric.Modification(None)],
    )
    
    adversarials, result = attack_eval.eval(
        test_dataset, 
        visualize=True,
    )

    dump_adv_examples(
        f'data/adv_examples/attack_base_{dataset_name}_{model_type}_{attacker_name}.json', 
        adversarials
    )

    our_result = Result(adversarials)
    our_result.print_stats()

    print("Attack: " + attacker_name + " ends")


# In[ ]:




