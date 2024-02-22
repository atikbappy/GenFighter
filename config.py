CONFIG = {
    "model_type": 'roberta-base', # 'bert-base-uncased' | 'roberta-base'
    "max_len": {
        "sst": 128,
        "agnews": 128,
        "imdb": 256,
    },
    "paraphraser_max_len": {
        "sst": 128,
        "agnews": 128,
        "imdb": 512,
    },
    "output_dim": {
        "sst": 2,
        "agnews": 4,
        "imdb": 2,
    },
    "num_of_para": {
        "sst": 12,
        "agnews": 15,
        "imdb": 12,
    },
    "paraphraser_name": 'Vamsi/T5_Paraphrase_Paws', # 'Vamsi/T5_Paraphrase_Paws' | 'humarin/chatgpt_paraphraser_on_T5_base'
    "paraphraser_type": 'google-paws', # 'google-paws' | 'chat-gpt'
    "use_paraphraser_seed": True,
    "version": "v1",
    "use_paraphrases_to_train_anomlay": {
        "sst": True,
        "agnews": False,
        "imdb": True,
    },
    "paraphraser_batch_size": {
        "sst": 32,
        "agnews": 32,
        "imdb": 16,
    },
    'anomaly_threshold': {
        "sst": 1314.8342, # 10th percentile
        "agnews": None, # 5th percentile
        "imdb": None, # 10th percentile
    },
}


ATTACK_CONFIG = {
    "attacker_names": ['PWWS', 'TextFooler', 'BertAttack'], # 'TextFooler' | 'BertAttack' | 'PWWS' | 'BAE'
    "test_split_starts": 0,
    "test_split_ends": 1000, # None will be full. If none then comment from code.
    "adv_file_post_fix": "0_1000",
    "use_top_k_weighted_mean": False,
    "use_trained_paraphraser": False,
    "save_adv_examples": True, 
    "visualize_attack": True,
    "test_seed": 769,
    "round_settings": {
        "sst": {  
            "num_of_round": 5,
            "TOP_K": 3,
        },
        "agnews": {
            "num_of_round": 5,
            "TOP_K": 3,
        },
        "imdb": {
            "num_of_round": 1,
            "TOP_K": 1,
        },
    }
}
