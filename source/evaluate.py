import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
from source.constants import EXP_DIR
from source.metrics import MAP_at_k, accuracy_at_1, accuracy_at_k_at_top_gold_1, flatten, normalize, precision_metrics_at_k, remove_word_from_list
from source.resources import get_last_experiment_dir
from source.generate import generate
from pytorch_lightning import seed_everything
from source.model import MyModel
from source.preprocessor import Preprocessor
from transformers import set_seed
import pandas as pd
from source.helper import log_params, log_stdout, get_experiment_dir, get_device
import json
import torch
from source.helper import count_line, log_stdout, unique, write_lines
from source import wandb_config
import wandb 
import json 
import gc 


def load_model(model_dirname=None):

    if model_dirname is None:  # default
        model_dir = get_last_experiment_dir()
    else:
        model_dir = EXP_DIR / model_dirname
    
    params = json.load((model_dir / "params.json").open('r'))
    seed_everything(params['seed'])
    # set_seed(params['seed'])
    
    best_model_path = Path(json.load((model_dir / 'best_model.json').open('r'))['best_model_path'])
    best_model_path = model_dir / best_model_path.name
    
    params['output_dir'] = model_dir
    
    print("Model dir: ", model_dir)
    print('check_point:', best_model_path)
    print("loading model...")
    
    
    checkpoint = MyModel.load_from_checkpoint(checkpoint_path=best_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = checkpoint.model.to(device)
    model.eval()
    return model, checkpoint.tokenizer, params


def evaluate_on(data, dataset, features_kwargs, phase, lang, model_dirname=None):
    
    model, tokenizer, params = load_model(model_dirname)
    seed_everything(params['seed'])
    # set_seed(params['seed'])
    
    params['eval_features'] = features_kwargs
    params['lang'] = lang 
    params['output_dir'] = Path(params['output_dir'])
    
    
    if 'mt5' in params['model_name']:
        tokenizer.src_lang = f'{lang}_XX'
        tokenizer.tgt_lang = f'{lang}_XX'
    
    max_len = int(params['max_seq_length'])

    os.environ['WANDB_API_KEY'] = wandb_config.WANDB_API_KEY
    os.environ['WANDB_MODE'] = wandb_config.WANDB_MODE
    wandb.init(project=wandb_config.WANDB_PROJECT_NAME, name=f"{params['output_dir'].stem}_Eval", job_type='Evaluate', config=params)
    

    output_dir = params['output_dir'] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)
    

    values = [f'{features_kwargs[key]["target_ratio"]:.2f}' for key in features_kwargs.keys()]    
    values = '_'.join(values)

    num_file = 0
    output_score_filepath = output_dir / f"{dataset}.{phase}.{values}.log.{num_file}.txt" 
    while output_score_filepath.exists():
        num_file += 1
        output_score_filepath = output_dir / f"{dataset}.{phase}.{values}.log.{num_file}.txt" 
    
    pred_filepath = output_dir / f'{dataset}.{phase}_{values}.tsv'
    pred_scores_filepath = output_dir / f'scores_{dataset}.{phase}_{values}.csv'
    print(pred_filepath)
    if pred_filepath.exists() and count_line(pred_filepath) >= len(data):
        print("File is already processed.")
    else:
        with log_stdout(output_score_filepath):
            complex_words = []
            list_pred_candidates = []
            list_gold_candidates = []
            list_sorted_gold_candidates = []
            
            pred_sents_list = []
            output = []
            logs = []
            for i in range(len(data)):
                row = data.iloc[i]
                complex_word = row["complex_word"]
                complex_words.append(complex_word)
                gold_candidates = json.loads(row['candidates'])
                list_gold_candidates.append(unique(flatten(gold_candidates)))
                list_sorted_gold_candidates.append(gold_candidates)
                
                source = row['source']
                pred_sents, pred_candidates = generate(source, model, tokenizer, max_len)
                pred_sents_list.append(pred_sents)
                
                # if verbose:
                print(f'{i}/{len(data)}', '='*80)
                print(source)
                # print(f'Unique candidates: {unique(gold_candidates)}')
                # print('\n'.join(pred_sents))
                
                pred_candidates = remove_word_from_list(complex_word, pred_candidates) # remove candidates the same as complex word
                pred_candidates = pred_candidates[:10] # limit it to max of 10
                print(f'Gold candidates: ', gold_candidates)
                print(f'Complex word: {complex_word}')
                print(f'Predicted candidates: ', pred_candidates)
                print('Predicted sentences:')
                print('\n'.join(pred_sents))
                
                list_pred_candidates.append(pred_candidates)
                output.append(f'{row["text"]}\t{row["complex_word"]}\t' + '\t'.join(pred_candidates))
            
            write_lines(output, pred_filepath)
            
            
            print("Features: ")  
            for key, val in features_kwargs.items():
                print(f'{key:<15}', ':', val['target_ratio'])              
                            
            to_save_data = {}
            
            # Acc@1
            value = accuracy_at_1(list_pred_candidates, list_gold_candidates)
            value = normalize(value)
            log_label = f'ACC@1'
            print(f'{log_label:>30}: {value}')
            logs.append(f'{log_label:>30}:\t{value}')
            wandb.log({f'AC@1': value})
            to_save_data[log_label] = value
            
            
            # compute accuracy@1@Top1
            for k in [1, 2, 3]:
                value = accuracy_at_k_at_top_gold_1(list_pred_candidates, list_sorted_gold_candidates, k)
                value = normalize(value)    
                print('='*20, f' Accuracy at k:{k} at top gold 1 ', '='*20)
                log_label = f'ACC@{k}@Top1'
                print(f'{log_label:>30}: {value}')
                logs.append(f'{log_label:>30}:\t{value}')
                wandb.log({f'ACC@{k}@Top1': value})
                to_save_data[log_label] = value
                
            # compute MAP
            for k in [1, 3, 5, 10]:
                value = MAP_at_k(list_pred_candidates, list_gold_candidates, k)
                value = normalize(value)
                print('='*20, f' MAP at k:{k} at top gold 1 ', '='*20)
                log_label = f'MAP@{k}'
                print(f'{log_label:>30}: {value}')
                logs.append(f'{log_label:>30}:\t{value}')
                wandb.log({f'MAP@{k}': value})
                to_save_data[log_label] = value
            
            for k in [1, 3, 5, 10]:
                print('='*20, f' Precision metrics at k:{k} ', '='*20)
                scores = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k)
                print(f'\n@{k}')
                for key, value in scores.items():
                    value = normalize(value)
                    log_label = f'{key}@{k}'
                    print(f'{log_label:25}:\t{value}') 
                    logs.append(f'{log_label:>30}:\t{value}')
                    wandb.log({f'{key}@{k}': f'{value}'})
                    to_save_data[log_label] = value
            
            
            pd.DataFrame([to_save_data]).to_csv(pred_scores_filepath, index=False)
                
            print('='*80) 
            logs = sorted(logs)
            for log in logs:
                print(log)
            
    wandb.finish()
    device = get_device()
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

model, tokenizer, params = None, None, None

def evaluation_pruning(data, dataset, features_kwargs, phase, lang, model_dirname=None):
    global model, tokenizer, params
    
    if model == None and tokenizer == None and params == None:
        model, tokenizer, params = load_model(model_dirname)
        seed_everything(params['seed'])
        
        params['eval_features'] = features_kwargs
        params['lang'] = lang 
        params['output_dir'] = Path(params['output_dir'])
        
        
        if 'mt5' in params['model_name']:
            tokenizer.src_lang = f'{lang}_XX'
            tokenizer.tgt_lang = f'{lang}_XX'
    
    max_len = int(params['max_seq_length'])
    output_dir = params['output_dir'] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)
    

    values = [f'{features_kwargs[key]["target_ratio"]:.2f}' for key in features_kwargs.keys()]    
    values = '_'.join(values)

    num_file = 0
    output_score_filepath = output_dir / f"{dataset}.{phase}_{values}.log.{num_file}.txt" 
    while output_score_filepath.exists():
        num_file += 1
        output_score_filepath = output_dir / f"{dataset}.{phase}.{values}.log.{num_file}.txt" 
    
    pred_filepath = output_dir / f'{dataset}.{phase}_{values}.tsv'
    pred_scores_filepath = output_dir / f'scores_{dataset}.{phase}_{values}.csv'
    print(pred_filepath)

    with log_stdout(output_score_filepath):
        complex_words = []
        list_pred_candidates = []
        list_gold_candidates = []
        list_sorted_gold_candidates = []
        
        pred_sents_list = []
        output = []
        logs = []
        for i in range(len(data)):
            row = data.iloc[i]
            complex_word = row["complex_word"]
            complex_words.append(complex_word)
            gold_candidates = json.loads(row['candidates'])
            list_gold_candidates.append(unique(flatten(gold_candidates)))
            list_sorted_gold_candidates.append(gold_candidates)
            source = row['source']
            pred_sents, pred_candidates = generate(source, model, tokenizer, max_len)
            pred_sents_list.append(pred_sents)
            
            # if verbose:
            print(f'{i}/{len(data)}', '='*80)
            print(source)
            # print(f'Unique candidates: {unique(gold_candidates)}')
            # print('\n'.join(pred_sents))
            
            pred_candidates = remove_word_from_list(complex_word, pred_candidates) # remove candidates the same as complex word
            pred_candidates = pred_candidates[:10] # limit it to max of 10
            print(f'Gold candidates: ', gold_candidates)
            print(f'Complex word: {complex_word}')
            print(f'Predicted candidates: ', pred_candidates)
            print('Predicted sentences:')
            print('\n'.join(pred_sents))
            
            list_pred_candidates.append(pred_candidates)
            output.append(f'{row["text"]}\t{row["complex_word"]}\t' + '\t'.join(pred_candidates))
        
        write_lines(output, pred_filepath)
        
        
        print("Features: ")  
        for key, val in features_kwargs.items():
            print(f'{key:<15}', ':', val['target_ratio'])              
                        
        to_save_data = {}
        
        # Acc@1
        value = accuracy_at_1(list_pred_candidates, list_gold_candidates)
        value = normalize(value)
        log_label = f'ACC@1'
        print(f'{log_label:>30}: {value}')
        logs.append(f'{log_label:>30}:\t{value}')
        # wandb.log({f'AC@1': value})
        to_save_data[log_label] = value
        
        acc1top1 = 0
        # compute accuracy@1@Top1
        for k in [1, 2, 3]:
            value = accuracy_at_k_at_top_gold_1(list_pred_candidates, list_sorted_gold_candidates, k)
            if k == 1: 
                acc1top1 = value
                
            value = normalize(value)    
                
            print('='*20, f' Accuracy at k:{k} at top gold 1 ', '='*20)
            log_label = f'ACC@{k}@Top1'
            print(f'{log_label:>30}: {value}')
            logs.append(f'{log_label:>30}:\t{value}')
            # wandb.log({f'ACC@{k}@Top1': value})
            to_save_data[log_label] = value
            
        # compute MAP
        for k in [1, 3, 5, 10]:
            value = MAP_at_k(list_pred_candidates, list_gold_candidates, k)
            value = normalize(value)
            print('='*20, f' MAP at k:{k} at top gold 1 ', '='*20)
            log_label = f'MAP@{k}'
            print(f'{log_label:>30}: {value}')
            logs.append(f'{log_label:>30}:\t{value}')
            # wandb.log({f'MAP@{k}': value})
            to_save_data[log_label] = value
        
        for k in [1, 3, 5, 10]:
            print('='*20, f' Precision metrics at k:{k} ', '='*20)
            scores = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k)
            print(f'\n@{k}')
            for key, value in scores.items():
                value = normalize(value)
                log_label = f'{key}@{k}'
                print(f'{log_label:25}:\t{value}') 
                logs.append(f'{log_label:>30}:\t{value}')
                # wandb.log({f'{key}@{k}': f'{value}'})
                to_save_data[log_label] = value
        
        pd.DataFrame([to_save_data]).to_csv(pred_scores_filepath, index=False)
            
        print('='*80) 
        logs = sorted(logs)
        for log in logs:
            print(log)
            
    # wandb.finish()
    device = get_device()
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    return acc1top1
    