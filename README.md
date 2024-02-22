## Settings
Global settings are configured under `config.py`

## Requirements
`Python 3.9`

Install `requirements.txt`

```
pip install -r requirements.txt
```

Install customized version `OpenAttack` manually by 

```
cd OpenAttack
python setup.py install
```

Original `OpenAttack` source code: https://github.com/thunlp/OpenAttack

## Training

### Training target model with paraphrases

`python train_gf_with_para.py --dataset_name sst`

### Training GMM model

`python train_gf_anomaly.py --dataset_name sst`

Grab percentile value and put under:

`config.py/CONFIG/anomaly_threshold`

## Attack

The following command will evaluate GenFighter against three adversarial attacks mentioned on our paper

`python attack_gf.py --dataset_name sst`

The results can be evaluated later using:

`python evaluate_results.py --dataset_name sst`

## Training and attacking target model without any defense
```
python train_base.py --dataset_name sst
python attack_base.py --dataset_name sst
```


