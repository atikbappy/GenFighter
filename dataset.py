from datasets.utils import disable_progress_bar
disable_progress_bar()

from config import CONFIG
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
import torch
import numpy as np
import pathlib
import random

BASE_PATH = str(pathlib.Path(__file__).parent.resolve())

def truncate_text(instance, column_name, max_words):
    words = instance[column_name].split()
    if len(words) > max_words:
        instance[column_name] = ' '.join(words[:max_words])
    return instance

def load_nlp_dataset(dataset=None):
    data_files={'train': BASE_PATH + '/data/{}_train.csv'.format(dataset),
                'validation': BASE_PATH + '/data/{}_val.csv'.format(dataset),
                'test': BASE_PATH + '/data/{}_test.csv'.format(dataset)}
    train_dataset = load_dataset('csv', data_files=data_files, split='train', ignore_verifications=False, name=dataset)
    eval_dataset = load_dataset('csv', data_files=data_files, split='validation', ignore_verifications=False, name=dataset)
    test_dataset = load_dataset('csv', data_files=data_files, split='test', ignore_verifications=False, name=dataset)
    
    # max_words = CONFIG["truncation_length"][dataset]
    # train_dataset = train_dataset.map(lambda instance: truncate_text(instance, "text", max_words))
    # eval_dataset = eval_dataset.map(lambda instance: truncate_text(instance, "text", max_words))
    # test_dataset = test_dataset.map(lambda instance: truncate_text(instance, "text", max_words))
    
    return train_dataset, eval_dataset, test_dataset

    
def pad_seq(seq, max_batch_len: int, pad_value: int):
    return seq + (max_batch_len - len(seq)) * [pad_value]

def collate_batch(batch, tokenizer, device) :
    batch_inputs = list()
    batch_attention_masks = list()
    labels = list()
    max_size = max([len(ex['input_ids']) for ex in batch])
    for item in batch:
        batch_inputs += [pad_seq(item['input_ids'], max_size, tokenizer.pad_token_id)]
        batch_attention_masks += [pad_seq(item['attention_mask'], max_size, 0)]
        if "label" in item:
            labels.append(item['label'])
    return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long).to(device),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long).to(device),
            "labels": torch.tensor(labels, dtype=torch.long).to(device)}


# Used duing the attack. Usually batch_size = 1, shuffle was added later
def prepare_single_bert(texts, tokenizer, batch_size=1, max_len=64, device='cpu', shuffle=False):
    def encode(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_len)
    my_dict = {"text": texts}
    dataset = Dataset.from_dict(my_dict)
    dataset = dataset.map(encode, batched=False)
    data_iter = DataLoader(
                dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                collate_fn=lambda p: collate_batch(p, tokenizer, device),
                drop_last=False,
            )
    return data_iter


def prepare_dataset_bert(model, dataset_name, batch_size=32, max_len=64, device='cpu'):
    train_dataset, eval_dataset, test_dataset = load_nlp_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model)

    def seed_worker(worker_id):
        print(torch.initial_seed())
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def encode(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_len)

    train_dataset = train_dataset.map(encode, batched=True)
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

    eval_dataset = eval_dataset.map(encode, batched=True)
    eval_dataset = eval_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

    test_dataset = test_dataset.map(encode, batched=True)
    test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    # import pdb; pdb.set_trace()

    train_iter = DataLoader(
                    train_dataset,
                    shuffle=True,
                    batch_size=batch_size,
                    collate_fn=lambda p: collate_batch(p, tokenizer, device),
                    worker_init_fn=seed_worker,
                    drop_last=False,
                )

    val_iter = DataLoader(
                eval_dataset,
                shuffle=False,
                batch_size=batch_size,
                collate_fn=lambda p: collate_batch(p, tokenizer, device),
                worker_init_fn=seed_worker,
                drop_last=False,
            )

    test_iter = DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=batch_size,
                collate_fn=lambda p: collate_batch(p, tokenizer, device),
                worker_init_fn=seed_worker,
                drop_last=False,
            )

    return train_iter, val_iter, test_iter, tokenizer

