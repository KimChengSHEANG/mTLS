from pathlib import Path;import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))# -- fix path --

from source.constants import EXP_DIR, REPO_DIR, Dataset, Language
from source.resources import get_last_experiment_dir, load_dataset, split_data, split_data_train_valid_test
from source.preprocessor import Preprocessor
from source.evaluate import evaluate_on
import torch
import json 
import pandas as pd 
import os



model_dir = 'exp_1679619309663310'
phase = 'test'
languages = ['en', 'es', 'pt'] # ['en'], ['es'], ['pt']
use_bert_pred_candidates = True

features_kwargs = {'en':{
                            'CandidateRanking': {'target_ratio': 1.00},
                            'WordLength': {'target_ratio': 0.80},
                            'WordRank': {'target_ratio': 0.80},
                            'WordSyllable':{'target_ratio': 0.80},
                            'SentenceSimilarity':{'target_ratio': 1.00},
                        },
                    'es':{
                            'CandidateRanking': {'target_ratio': 1.00},
                            'WordLength': {'target_ratio': 0.80},
                            'WordRank': {'target_ratio': 0.80},
                            'WordSyllable':{'target_ratio': 0.80},
                            'SentenceSimilarity':{'target_ratio': 1.00},
                        },
                    'pt': {
                            'CandidateRanking': {'target_ratio': 1.00},
                            'WordLength': {'target_ratio': 0.80},
                            'WordRank': {'target_ratio': 0.80},
                            'WordSyllable':{'target_ratio': 0.80},
                            'SentenceSimilarity':{'target_ratio': 1.00},
                        },              
                   }

all_test_set = {}
for lang in languages:
    
    preprocessor = Preprocessor(features_kwargs[lang], lang)
    data = pd.read_csv(EXP_DIR / f'{model_dir}/data/tsar-{lang}.{phase}.csv')
    data = preprocessor.preprocess_valid_or_test_set(data, use_bert_pred_candidates)
    all_test_set[lang] = data

for lang in languages:
    test_set = all_test_set[lang]
    dataset = f'tsar-{lang}'
    evaluate_on(test_set, dataset, features_kwargs[lang], phase=phase, lang=lang, model_dirname=model_dir)


