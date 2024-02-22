import json
import math
import random
import time

import numpy as np
import tensorflow as tf
import torch
from config import CONFIG
from dataset import prepare_single_bert
from model import BertClassifierDARTS
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AdamW, AutoModelForSeq2SeqLM, AutoTokenizer,
                          RobertaModel, RobertaTokenizer,
                          get_linear_schedule_with_warmup)
from utils import evaluate_batch_single, evaluate_without_attack, get_preds

random.seed(12)
torch.manual_seed(12)
tf.random.set_seed(12)
np.random.seed(12)

torch.cuda.manual_seed(12)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_paraphrases(
    model_name, 
    sentences, 
    num_of_para, 
    device, 
    load_path=None,  
    max_len=128, 
    cached_model=None, 
    cached_tokenizer=None,
    model_type='google-paws', # 'google-paws' | 'chat-gpt',
    batch_size=32,
    disable_tqdm=True,
    seed=None,
):
    # torch.manual_seed(12)
    if not isinstance(sentences, list):
        raise ValueError("Sentences must be list")
    # start_time = time.time()
    if cached_model:
        model = cached_model
        tokenizer = cached_tokenizer
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        if load_path:
            model.load_state_dict(torch.load(load_path))
    
    res = list()
    for i in tqdm(range(0, len(sentences), batch_size), disable=disable_tqdm):
        res.extend(
            get_batch_paraphrases(
                tokenizer,
                model, 
                sentences[i:i + batch_size], 
                num_of_para, 
                max_len, 
                device,
                model_type,
                seed,
            )
        )
    
    if not cached_model:
        del model
        del tokenizer
        torch.cuda.empty_cache()
    # print("--- %s seconds ---" % (time.time() - start_time))
    return res

def get_batch_paraphrases(
    tokenizer, 
    model, 
    sentences, 
    num_of_para, 
    max_len, 
    device,
    model_type,
    seed,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
):
    input = ["paraphrase: " + s + " </s>" for s in sentences]
    
    encoding = tokenizer(input, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    if CONFIG["use_paraphraser_seed"]:
        torch.manual_seed(seed or 12)

    if model_type == 'google-paws':
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_masks,
            max_length=max_len,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=num_of_para,
        )
    elif model_type == 'chat-gpt':
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_masks, 
            temperature=temperature, 
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_of_para, 
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_of_para, 
            num_beam_groups=num_of_para,
            max_length=max_len, 
            diversity_penalty=diversity_penalty
        )

    paraphrased_texts = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    paraphrased_texts = [p.lower() for p in paraphrased_texts]
    res = [
        paraphrased_texts[i: i + num_of_para] 
        for i in range(0, len(paraphrased_texts), num_of_para)
    ]
    return res


class BaseModelDataset(Dataset):
    def __init__(
        self, 
        tokenizer,
        sentences,
        max_len,
        device,
        labels=None  # Sometime it will have labels and sometime it will not
    ):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_len = max_len
        self.device = device
        self.labels = labels
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer.batch_encode_plus(
            [self.sentences[idx]], 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True,
            pad_to_max_length=True,
            max_length=self.max_len,
        )
        
        inputs['input_ids'] = inputs['input_ids'].squeeze().to(self.device, dtype=torch.long)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze().to(self.device, dtype=torch.long)
        
        res = {
            "input_ids": inputs['input_ids'],
            "attention_mask":  inputs['attention_mask'],
        }
        if self.labels:
            res['labels'] = torch.tensor(self.labels[idx], dtype=torch.long).to(self.device)
        return res

def get_confidences(
    sentences, 
    sub_list_size, 
    device, 
    dataset_name,
    model_type,
    load_path=None, 
    cached_model=None, 
    cached_tokenizer=None,
):
    max_len=CONFIG["max_len"][dataset_name]
    batch_size=32

    if cached_model:
        model = cached_model
        tokenizer = cached_tokenizer
    else:
        model = BertClassifierDARTS(
            model_type=model_type, 
            freeze_bert=False, 
            output_dim=CONFIG["output_dim"][dataset_name], 
            ensemble=0, 
            device=device
        )
        if load_path:
            model.load_state_dict(torch.load(load_path))
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    model.eval()
    dataset = BaseModelDataset(
        tokenizer, sentences, max_len, device
    )
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    preds = get_preds(model, data_iter)
    # preds = softmax(preds, axis=1) # The get_preds is alredy doing softmax here
    preds = preds.tolist()

    if not cached_model:
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    if sub_list_size:
        return [
            preds[i : i + sub_list_size] 
            for i in range(0, len(preds), sub_list_size)
        ]
    else:
        return preds

def flatten_double_list(paraphrases):
    temp_para = list()
    for p in paraphrases:
        temp_para.extend(p)
    return temp_para

def val_accuracy(data_iter, preds):
    preds = np.argmax(preds, axis=1)
    labels = [a['label'] for a in data_iter.dataset]
    acc = accuracy_score(labels, preds)
    return acc


class ScoringDataset(Dataset):
    def __init__(
        self, 
        tokenizer,
        original_sentences, 
        paraphrases, 
        confidence_values, 
        device,
        max_len,
        labels=None,
    ):
        self.tokenizer = tokenizer
        self.original_sentences = original_sentences
        self.paraphrases = paraphrases
        self.confidences = confidence_values
        self.device = device
        self.max_len = max_len
        self.labels = labels
        
    def __len__(self):
        return len(self.original_sentences)
    
    def __getitem__(self, idx):
        paraphrase_texts = [
            self.original_sentences[idx] + self.tokenizer.sep_token + par for par in self.paraphrases[idx]
        ]
        inputs = self.tokenizer.batch_encode_plus(
            paraphrase_texts, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True,
            pad_to_max_length=True,
            max_length=self.max_len,
        )
        
        inputs['input_ids'] = inputs['input_ids'].to(self.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
        
        confidences = torch.tensor(self.confidences[idx], dtype=torch.float32).to(self.device)
        label = torch.tensor(self.labels[idx], dtype=torch.long).to(self.device) if self.labels else torch.empty(1)
        
        return inputs, confidences, label, self.original_sentences[idx]


class T5ParaDataSet(Dataset):
    def __init__(
        self, 
        tokenizer, 
        source_text, 
        target_text,
        device,
        max_len,
    ):
        self.tokenizer = tokenizer
        self.target_text = source_text
        self.source_text = target_text
        self.device = device
        self.max_len = max_len

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = "paraphrase: " + str(self.source_text[index]) + " "
        target_text = str(self.target_text[index]) + " "

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
            "target_mask": target_mask.to(dtype=torch.long),
        }


from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)

NUM_OF_PARA = 7

class BaseModel:
    tokenizer = None
    model = None
    optimizer = None
    tokenizer_class = None
    model_class = None
    num_of_epoch = None
    MAX_LENGTH = 128
    
    def __init__(self, tokenizer_class, model_class, device, print_model=False, load_path=None):
        self.tokenizer_class = tokenizer_class
        self.model_class = model_class
        self.device = device
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name)
        self.model = self.model_class.from_pretrained(self.model_name).to(self.device)
        
        if load_path:
            self.model.load_state_dict(
                torch.load(load_path)
            )

        if print_model:
            print(self.model)

    def init_logger(self):
        self.training_logger = Table(
            Column("Epoch", justify="center"),
            Column("Loss", justify="center"),
            Column("Accuracy", justify="center"),
            title="Training Status",
            pad_edge=False,
            box=box.ASCII,
        )

    def save_model(self, path, commit=False):
        if commit:
            torch.save(self.model.state_dict(), path)
        print("Model saved. (Commit=" + str(commit) + ")")

    def model_destroy(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()



class FineTuneScoringModel(BaseModel):
    model_name = 'roberta-base'
    linear_layer = None
    BATCH_SIZE = 8
    NUM_OF_EPOCH = 5
    result = None
    NUMBER_OF_PARA = NUM_OF_PARA
    LEARNING_RATE = 1e-5
    scores = None
    
    def __init__(self, device, load_path=None):
        super().__init__(RobertaTokenizer, RobertaModel, device, load_path=load_path)
        self.linear_layer = torch.nn.Linear(
            self.MAX_LENGTH * self.model.config.hidden_size, 1
        ).to(device)
        
        self.init_layers()

    def init_layers(self):
        # Fine-tuning parameters
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.LEARNING_RATE)

    def check_classification(self, temp_prob, orig_label):
        temp_label = torch.argmax(torch.tensor(temp_prob, dtype=torch.float32), dim=1)
        return temp_label == orig_label

    def get_relevance_score(self, batch, allow_grad=False):
        with torch.set_grad_enabled(allow_grad):
            inputs, confidences, label, orig_sentences = batch
            batch_size = label.shape[0]
            
            # The RobertaModel and most other transformer models expect input in 2D tensors
            inputs['input_ids'] = inputs['input_ids'].view(batch_size * self.NUMBER_OF_PARA, -1)
            inputs['attention_mask'] = inputs['attention_mask'].view(batch_size * self.NUMBER_OF_PARA, -1)
            
            # with torch.no_grad():
            base_outputs = self.model(**inputs) # requires both dict_keys(['input_ids', 'attention_mask'])
            last_hidden_state = base_outputs[0] # should have taken the first element of lhs because it's the [CLS]
            
            logits = last_hidden_state.view(batch_size, self.NUMBER_OF_PARA, self.MAX_LENGTH, -1)
            linear_logits = self.linear_layer(logits.view(batch_size, self.NUMBER_OF_PARA, -1))
            soft_logits = torch.softmax(linear_logits, dim=1)
            self.scores.extend(soft_logits.tolist())
            
            weighted_probs = soft_logits * confidences
            relevance_score = torch.sum(weighted_probs, dim=1)
    
            return relevance_score

    def train(self, dataloader):
        self.model.train()
        self.init_logger()
        
        for epoch in range(self.NUM_OF_EPOCH):
            total_loss = 0.0
            self.scores, self.result = list(), list()
            
            for batch in tqdm(dataloader):
                inputs, confidences, label, orig_sentences = batch
                relevance_score = self.get_relevance_score(batch, allow_grad=True)
                
                loss_fn = torch.nn.CrossEntropyLoss()  # Cross-Entropy loss for softmax (as confidence). If we had used sigmoid AS confidence then BCE
                loss = loss_fn(relevance_score, label)
                total_loss += loss.item()
                
                self.result.extend(
                    self.check_classification(relevance_score, label).tolist()
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            avg_loss = total_loss / len(dataloader)
            accuracy = sum(self.result) / len(self.result)

            self.training_logger.add_row(str(epoch + 1) + " / " + str(self.NUM_OF_EPOCH), str(f"{avg_loss:.4f}"), str(f"{accuracy:.4f}"))
            console.print(self.training_logger)

        return avg_loss, accuracy, self.scores  # last epoch result


    def eval(self, dataloader):
        self.model.eval()
        self.init_logger()
        self.scores, self.result = list(), list()
            
        for batch in tqdm(dataloader):
            inputs, confidences, label, orig_sentences = batch
            relevance_score = self.get_relevance_score(batch)
            self.result.extend(
                self.check_classification(relevance_score, label).tolist()
            )
           
        accuracy = sum(self.result) / len(self.result)

        self.training_logger.add_row(str("*"), str("*"), str(f"{accuracy:.4f}"))
        console.print(self.training_logger)

        return accuracy, self.scores


class FineTuneT5Paraphraser(BaseModel):
    model_name = 'Vamsi/T5_Paraphrase_Paws'
    dataloader = None
    BATCH_SIZE = 16
    NUM_OF_EPOCH = 5
    LEARNING_RATE = 1e-5
    NUMBER_OF_PARA = math.floor(FineTuneScoringModel.NUMBER_OF_PARA / 2) # watchout
    
    def __init__(self, device, load_path=None):
        super().__init__(AutoTokenizer, AutoModelForSeq2SeqLM, device, print_model=False, load_path=load_path)
        self.init_layers()
        

    def init_layers(self):
        # Fine-tuning parameters
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.LEARNING_RATE)
        

    def train(self, dataloader):
        self.model.train()
        self.init_logger()
        
        for epoch in range(self.NUM_OF_EPOCH):
            total_loss = 0.0
            for _, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
                # y = data["target_ids"].to(self.device, dtype=torch.long)
                # y_ids = y[:, :-1].contiguous()
                # lm_labels = y[:, 1:].clone().detach()
                # lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
                # ids = data["source_ids"].to(self.device, dtype=torch.long)
                # mask = data["source_mask"].to(self.device, dtype=torch.long)
        
                # outputs = self.model(
                #     input_ids=ids,
                #     attention_mask=mask,
                #     decoder_input_ids=y_ids,
                #     labels=lm_labels,
                #     labels=y,
                # )

                input_ids = data["source_ids"].to(self.device, dtype=torch.long)
                source_mask = data["source_mask"].to(self.device, dtype=torch.long)
                labels = data["target_ids"].to(self.device, dtype=torch.long)
                target_mask = data["target_mask"].to(self.device, dtype=torch.long)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=source_mask,
                    labels=labels,
                    decoder_attention_mask=target_mask,
                )
                
                
                loss = outputs.loss
                total_loss += loss.item()
        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(dataloader)
            self.training_logger.add_row(str(epoch + 1) + " / " + str(self.NUM_OF_EPOCH), str(f"{avg_loss:.4f}"), str("*"))
            console.print(self.training_logger)


def dump_adv_examples(path, adversarials):
    for adv in adversarials:
        if isinstance(adv[1], np.ndarray):
            adv[1] = adv[1].tolist()
        if isinstance(adv[3], np.ndarray):
            adv[3] = adv[3].tolist()
    with open(path, "w") as f:
        json.dump(adversarials, f, indent=4)


def get_anomalous_scores(text_model, tokenizer, anomaly_model, texts, max_len, device, predict=False):
    # convert to embeddings
    X = np.empty((len(texts), 768), )
    index = 0
    data_iter = prepare_single_bert(
        texts, 
        tokenizer=tokenizer, 
        batch_size=1, 
        max_len=max_len,
        device=device
    )
    for batch in data_iter:
        seq = batch['input_ids']
        attn_masks = batch['attention_mask']
        cont_reps = text_model.bert_layer(input_ids=seq, attention_mask=attn_masks)
        lhs = cont_reps.last_hidden_state[:, 0]
        
        for logits in lhs:
            X[index] = logits.cpu().numpy()
            index += 1

    if predict:
        return anomaly_model.predict(X)
    return anomaly_model.score_samples(X) # lower values are more anomalous

