import sys

import joblib
import numpy as np
import OpenAttack as oa
import tensorflow as tf
import torch

from config import CONFIG, ATTACK_CONFIG
from dataset import load_nlp_dataset, prepare_single_bert
from helpers import flatten_double_list, get_anomalous_scores, get_confidences, get_paraphrases
from scipy.special import softmax
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import get_preds

rng = np.random.default_rng(12)


class InputTracker:
    def __init__(self, index, texts):
        self.index = index
        self.texts = texts
        self.completed = False


def pick_less_anomalous_text():
    pass


class AnomalyClassifier(oa.Classifier):
    def __init__(
        self, 
        model, 
        tokenizer, 
        dataset_name,
        model_type,
        clf,
        max_len,
        batch_size=1,
        device='cpu',
        paraphraser_seed=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.clf = clf
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device
        self.paraphraser_seed = paraphraser_seed
        
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["paraphraser_name"]).to(device)
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained(CONFIG["paraphraser_name"])

        if ATTACK_CONFIG["use_trained_paraphraser"]:
            folder = self.dataset_name + '_' + model_type
            self.paraphrase_model.load_state_dict(
                torch.load(f"./models/{folder}/paraphraser{CONFIG['version']}.pt")
            )

        self.ROUND = ATTACK_CONFIG["round_settings"][self.dataset_name]["num_of_round"]
        self.TOP_K = ATTACK_CONFIG["round_settings"][self.dataset_name]["TOP_K"]
        
        assert self.ROUND >= self.TOP_K, "ROUND must be greater or equal than TOP_K"

    def get_pred_prob(self, texts):
        probs = self.get_prob(texts)
        preds = probs.argmax(axis=1)
        return preds, probs
    
    def get_pred(self, texts):
        probs = self.get_prob(texts)
        return probs.argmax(axis=1)

    def get_normalized_anomaly_scores(self, anomaly_scores):
        if np.all(anomaly_scores == anomaly_scores[0]):
            return np.ones(anomaly_scores.shape[0]) / anomaly_scores.shape[0]

        normalized_anomaly_scores = (
            anomaly_scores - anomaly_scores.min()
        ) / (anomaly_scores.max() - anomaly_scores.min() + sys.float_info.epsilon)

        return normalized_anomaly_scores / normalized_anomaly_scores.sum()

    def get_prob(self, texts): # It's been called a lot from the attacker class. So, don't include extensive operations here.
        preds = np.empty([len(texts), CONFIG["output_dim"][self.dataset_name]])
        trackers = [InputTracker(i, [text]) for i, text in enumerate(texts)]

        for round in range(self.ROUND):
            to_paraphrase = []
            for track in filter(lambda t: not t.completed, trackers):
                to_paraphrase.extend(track.texts)

            paraphrases = get_paraphrases(
                CONFIG["paraphraser_name"],  
                to_paraphrase, 
                CONFIG["num_of_para"][self.dataset_name], 
                self.device, 
                max_len=CONFIG["paraphraser_max_len"][self.dataset_name],
                cached_model=self.paraphrase_model,
                cached_tokenizer=self.paraphrase_tokenizer,
                model_type=CONFIG["paraphraser_type"],
                batch_size=CONFIG["paraphraser_batch_size"][self.dataset_name],
                # seed=self.paraphraser_seed,
                # seed=rng.integers(1000),
            )
            paraphrases = flatten_double_list(paraphrases)
            
            sentences = list()
            start = 0
            for track in filter(lambda t: not t.completed, trackers):
                length = len(track.texts) * CONFIG["num_of_para"][self.dataset_name]
                sentences.extend(track.texts)
                sentences.extend(
                    paraphrases[start: start + length]
                )
                start += length
                
            scores = get_anomalous_scores(
                self.model, 
                self.tokenizer, 
                self.clf, 
                sentences, 
                self.max_len, 
                self.device
            )
            start = 0
            for track in filter(lambda t: not t.completed, trackers):
                length = len(track.texts) + len(track.texts) * CONFIG["num_of_para"][self.dataset_name]
                anomaly_scores = scores[start: start + length]
                higher_indexes = anomaly_scores.argsort()[::-1][:self.TOP_K]
                top_k_scores = anomaly_scores[higher_indexes]
                track.texts = np.array(sentences[start: start + length])[higher_indexes]
                track.texts = track.texts.tolist()

                # If all top scores are same
                if np.all(top_k_scores == top_k_scores[0]):
                    top_k_scores = np.array([top_k_scores[0]])
                    track.texts = [track.texts[0]]

                if np.all(top_k_scores >= CONFIG["anomaly_threshold"][self.dataset_name]) or round == self.ROUND - 1:
                    normalized_anomaly_scores = self.get_normalized_anomaly_scores(
                        top_k_scores
                    )
                    confidences = get_confidences(
                        track.texts,
                        None, 
                        self.device, 
                        self.dataset_name,
                        self.model_type,
                        cached_model=self.model, 
                        cached_tokenizer=self.tokenizer,
                    )
                    confidences = np.array(confidences)
                    preds[track.index] = normalized_anomaly_scores.dot(confidences)
                    track.completed = True  

                start += length

            if len(list(filter(lambda t: not t.completed, trackers))) == 0:
                break
            
        return preds