from pathlib import Path;import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))# -- fix path --


from source.constants import CACHE_DIR, REPO_DIR, RESOURCES_DIR, Dataset, Language
from source.resources import load_dataset, split_data, split_data_train_valid_test
from source.preprocessor import Preprocessor
from source.train import train_on
from source.helper import get_experiment_dir, get_device
from source.evaluate import evaluate_on
import torch
import json 
import pandas as pd 
import gc 
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


kwargs = dict(
    dataset='tsar',
    
    # mt5-large
    model_name='t5-large',
    train_batch_size=4,
    valid_batch_size=4,
    gradient_accumulation_steps=4,
    
    max_seq_length=210,
    learning_rate=1e-5, #3e-4 , 1e-5
    weight_decay=0.1,
    adam_epsilon=1e-8,
    warmup_steps=5,
    num_train_epochs=30,
    n_gpu=torch.cuda.device_count(),
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true  https://github.com/NVIDIA/apex
    opt_level='01', # 01, you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
    nb_sanity_val_steps=1,
    output_dir=get_experiment_dir(),
    save_top_k=1,
    early_stop_callback=True,
    earlystop_patience=4
)

train_features_kwargs = {
        'CandidateRanking': {'target_ratio': 1.00},
        'WordLength': {'target_ratio': 1.00},
        'WordRank': {'target_ratio': 1.00},
        'WordSyllable':{'target_ratio': 1.00},
        'SentenceSimilarity':{'target_ratio': 1.00}
    }
test_features_kwargs = {
        'CandidateRanking': {'target_ratio': 1.00},
        'WordLength': {'target_ratio': 1.00},
        'WordRank': {'target_ratio': 1.00},
        'WordSyllable':{'target_ratio': 1.00},
        'SentenceSimilarity':{'target_ratio': 1.00}
    }
languages = ['en']

topk = 10
use_mask_pred_candidates = False

output_dir = kwargs['output_dir'] / 'data'
output_dir.mkdir(parents=True, exist_ok=True)

all_train_set_processed = []
all_valid_set_processed = []
all_test_set = {}
print('Preprocessing...')
for lang in languages:
    dataset_name = f'tsar-{lang}'
    preprocessor = Preprocessor(train_features_kwargs, lang)
    if lang == 'en':
        data = pd.read_csv(RESOURCES_DIR / f'mask_pred_candidates/tsar-en_topk_{topk}_roberta-base.csv')
    elif lang == 'es':
        data = pd.read_csv(RESOURCES_DIR / f'mask_pred_candidates/tsar-es_topk_{topk}_PlanTL-GOB-ES|roberta-large-bne.csv')
    elif lang == 'pt':
        data = pd.read_csv(RESOURCES_DIR / f'mask_pred_candidates/tsar-pt_topk_{topk}_neuralmind|bert-large-portuguese-cased.csv')
        
    train_set, valid_set, test_set = split_data_train_valid_test(data, frac=0.7, seed=42)
    train_processed_out_file = CACHE_DIR / f'{dataset_name}.train.processed.csv' 
    train_set_processed = preprocessor.preprocess_train_set(train_set, use_mask_pred_candidates)
    
    
    valid_set = preprocessor.preprocess_valid_or_test_set(valid_set, use_mask_pred_candidates)

    preprocessor = Preprocessor(test_features_kwargs, lang)
    test_set = preprocessor.preprocess_valid_or_test_set(test_set, use_mask_pred_candidates)
    
    # save data
    data.to_csv(output_dir / f'{dataset_name}.csv', index=False)
    train_set.to_csv(output_dir / f'{dataset_name}.train.csv', index=False)
    valid_set.to_csv(output_dir / f'{dataset_name}.valid.csv', index=False)
    test_set.to_csv(output_dir / f'{dataset_name}.test.csv', index=False)
    train_set_processed.to_csv(output_dir / f'{dataset_name}.train.processed.csv', index=False)
    all_train_set_processed.append(train_set_processed)
    all_valid_set_processed.append(valid_set)
    all_test_set[lang] = test_set
    
all_train_set_processed = pd.concat(all_train_set_processed)
all_valid_set_processed = pd.concat(all_valid_set_processed)
all_train_set_processed = all_train_set_processed.sample(frac=1)
all_valid_set_processed = all_valid_set_processed.sample(frac=1)
all_train_set_processed.to_csv(output_dir / 'train.combined.csv', index=False)
all_valid_set_processed.to_csv(output_dir / 'valid.combined.csv', index=False)

if get_device() == 'cuda':
    torch.cuda.empty_cache()
gc.collect()
    
train_on(all_train_set_processed, all_valid_set_processed, kwargs)

# for lang in languages:
#     test_set = all_test_set[lang]
#     dataset = f'tsar-{lang}'
#     evaluate_on(test_set, dataset, test_features_kwargs, phase='test', lang=lang)


