from pathlib import Path;import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))# -- fix path --

from source.constants import EXP_DIR, REPO_DIR, RESOURCES_DIR, Dataset, Language
from source.resources import get_last_experiment_dir, load_dataset, split_data, split_data_train_valid_test
from source.preprocessor import Preprocessor
from source.evaluate import evaluate_on
import torch
import json 
import pandas as pd 
import os



model_dir = None #'exp_1679619309663310'
phase = 'test'
lang = 'en'
use_bert_pred_candidates = True

features_kwargs = {
        'CandidateRanking': {'target_ratio': 1.00},
        'WordLength': {'target_ratio': 0.80},
        'WordRank': {'target_ratio': 0.80},
        'WordSyllable':{'target_ratio': 0.80},
        'SentenceSimilarity':{'target_ratio': 1.00},
    }

preprocessor = Preprocessor(features_kwargs, lang)
benchls = pd.read_csv(RESOURCES_DIR / 'mask_pred_candidates/BenchLS_topk_10_bert-large-cased.csv')
lexmturk = pd.read_csv(RESOURCES_DIR / 'mask_pred_candidates/lex.mturk_topk_10_roberta-base.csv')
nnseval = pd.read_csv(RESOURCES_DIR / 'mask_pred_candidates/NNSeval_topk_5_bert-base-uncased.csv')

benchls = preprocessor.preprocess_valid_or_test_set(benchls, use_mask_pred_candidates=use_bert_pred_candidates)
lexmturk = preprocessor.preprocess_valid_or_test_set(lexmturk, use_mask_pred_candidates=use_bert_pred_candidates)
nnseval = preprocessor.preprocess_valid_or_test_set(nnseval, use_mask_pred_candidates=use_bert_pred_candidates)



evaluate_on(benchls, 'BenchLS', features_kwargs, phase=phase, lang=lang, model_dirname=model_dir)
evaluate_on(lexmturk, 'LexMTurk', features_kwargs, phase=phase, lang=lang, model_dirname=model_dir)
evaluate_on(nnseval, 'NNSeval', features_kwargs, phase=phase, lang=lang, model_dirname=model_dir)


