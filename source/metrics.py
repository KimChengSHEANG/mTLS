# -*- coding: utf-8 -*-

# Adopted from the official evaluation script for TSAR-2022 Shared Task on Lexical Simplification for English, Portuguese and Spanish.
# site: https://taln.upf.edu/pages/tsar2022-st/
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # fix path

import math
from collections import Counter, defaultdict
from pathlib import Path
from source.helper import safe_division, yield_lines
import pandas as pd 

def normalize(value):	
    factor = 10000	
    res = (math.floor(value * factor) / factor) * 100
    return f'{res:.2f}'
           
def unique(items): 
    # filter duplicate and preserve order
    return list(dict.fromkeys(items))

def remove_word_from_list(word, word_list):
    return [w for w in word_list if w.lower() != word.lower()]

def sort_candidates_by_ranking(candidates):
    # it is for NNSeval, BenchLS, ... 
    # candidates: [('parts', 1), ('component', 2), ('sections', 2), ...]
    
    dict_candidates = defaultdict(list)
    for word, rank in candidates:
        dict_candidates[rank].append(word.strip().lower())
    
    sorted_keys = sorted(dict_candidates.keys())
    return [dict_candidates[key] for key in sorted_keys]
    

def sort_candidates_by_frequency(candidates):
    ''' lex.mturk, tsar, 
    Return ranked candidates in groups: [['parts'], ['bits'], ['components'], ['component', 'sections', 'elements', 'part', 'information', 'items']]
    '''
    candidates = Counter(candidates).items()
    # [('parts', 40), ('component', 1), ('sections', 1), ('elements', 1), ('part', 1), ('components', 2), ('bits', 3), ('information', 1), ('items', 1)
    dict_candidates = defaultdict(list)
    for word, freq in candidates:
        dict_candidates[freq].append(word)
        
    sorted_keys = sorted(dict_candidates.keys(), reverse=True)
    return [dict_candidates[key] for key in sorted_keys]

def flatten(grouped_candidates):
    return [item for items in grouped_candidates for item in items]
    
def match(pred_candidate, gold_candidates):
    gold_candidates = list(gold_candidates)
    return pred_candidate in gold_candidates

def match_group(pred_candidates, gold_candidates):
    return any(match(pred, gold_candidates) for pred in pred_candidates)

def accuracy_at_1(list_pred_candidates, list_gold_candidates):
    # Accuracy Metric
    tp = 0
    total = 0

    for pred_candidates, gold_candidates in zip(list_pred_candidates, list_gold_candidates):
        if pred_candidates and len(pred_candidates[0]) > 0:
            if match(pred_candidates[0], gold_candidates):
                tp += 1
        total += 1

    return safe_division(tp, total)

def accuracy_at_k_at_top_gold_1(list_pred_candidates, list_sorted_gold_candidates, k):
    # Accuracy Metric
    tp = 0
    total = 0

    for pred_candidates, gold_candidates in zip(list_pred_candidates, list_sorted_gold_candidates):
        if pred_candidates and len(pred_candidates[:k]) > 0:
            if match_group(pred_candidates[0:k], gold_candidates[0]):
                tp += 1
        total += 1

    return safe_division(tp, total)

def precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k):

    # Precision
    precision = 0
    recall = 0
    f1 = 0

    running_precision = 0
    running_recall = 0

    potential_counts = 0
    potential = 0

    total = 0

    for pred_candidates, gold_candidates in zip(list_pred_candidates, list_gold_candidates):
        labels = pred_candidates[:k]
        if len(labels) > 0:
            acc_labels = [l for l in labels if match(l, gold_candidates)]
            acc_gold = [l for l in gold_candidates if match(l, labels)]

            if len(acc_labels) > 0:
                potential_counts += 1

            precision = safe_division(len(acc_labels), len(labels))
            recall = safe_division(len(acc_gold), len(gold_candidates))

            running_precision += precision
            running_recall += recall
        
        total += 1

    precision = safe_division(running_precision, total)
    recall = safe_division(running_recall, total)


    f1 = 0
    if (precision + recall) > 0:
        f1 = safe_division(2 * precision * recall, (precision + recall))

    if (potential_counts > 0):
        potential = safe_division(potential_counts, total)

    return {'precision': precision, 
            'recall': recall, 
            'f1': f1,
            'potential': potential
            }

# Mean Average Precision
# Parameters :
#  1. List of Binary Relevance Judgments e.g. [False, True, True, False, False]
#  2. K

def compute_local_MAP(list_gold_items_match, k):
    list_gold_items_match = list_gold_items_match[:k]
    AP = 0
    TruePositivesSeen = 0
    for index, item in enumerate(list_gold_items_match, start=1):
        if item == True:
            TruePositivesSeen += 1
            precision = safe_division(TruePositivesSeen, index)
            AP += precision

    return safe_division(AP, k)

def MAP_at_k(list_pred_candidates, list_gold_candidates, k):

    total_instances = 0
    MAP_global_accumulator = 0

    for pred_candidates, gold_candidates in zip(list_pred_candidates, list_gold_candidates):
        
        labels_relevance_judgements = [match(label, gold_candidates) for label in pred_candidates]
        MAP_local = compute_local_MAP(labels_relevance_judgements, k)
        MAP_global_accumulator += MAP_local
        total_instances += 1

    MAP = 0
    if (MAP_global_accumulator > 0):
        MAP = safe_division(MAP_global_accumulator, total_instances)
    return MAP


class Evaluator(object):

    def __init__(self, filter_complex_word=True):
        self.filter_complex_word = filter_complex_word
    
    def __load_data_with_index(self, filepath):
        self.sentences = []
        self.complex_words = []
        self.list_gold_candidates = []
        self.list_sorted_gold_candidates = []
        
        # load gold candidates
        for line in yield_lines(filepath):
            
            chunks = line.split('\t')
            self.sentences.append(chunks[0])
            complex_word = chunks[1]
            self.complex_words.append(complex_word)
            candidates = chunks[3:]
            candidates = [tuple(candidate.split(':')) for candidate in candidates] # reverse (num, word) => (word, num)
            candidates = [(word.strip().lower(), index) for index, word in candidates]
             
            if self.filter_complex_word:
                candidates = [w for w in candidates if w[0].lower() != complex_word.lower()]
                
            self.list_gold_candidates.append(unique([c[0] for c in candidates]))
            sorted_candidates = sort_candidates_by_ranking(candidates)
            self.list_sorted_gold_candidates.append(sorted_candidates)

    def __load_data(self, filepath):
        self.sentences = []
        self.complex_words = []
        self.list_gold_candidates = []
        self.list_sorted_gold_candidates = []
        
        # load gold candidates
        for line in yield_lines(filepath):
            
            chunks = line.split('\t')
            self.sentences.append(chunks[0])
            complex_word = chunks[1]
            self.complex_words.append(complex_word)
            candidates = chunks[2:]
            candidates = [word.strip().lower() for word in candidates]
            
            if self.filter_complex_word:
                candidates = remove_word_from_list(complex_word, candidates) 
                
            self.list_gold_candidates.append(unique(candidates))
            sorted_candidates = sort_candidates_by_frequency(candidates)
            self.list_sorted_gold_candidates.append(sorted_candidates)

    
    def load_gold_data(self, filepath):
        
        lines = yield_lines(filepath)
        next(lines)
        line = next(lines)
        chunks = line.split('\t')
        
        if chunks[2].isdigit():
            self.__load_data_with_index(filepath)
        else:
            self.__load_data(filepath)
        
    def load_predicted_data(self, filepath):
        # load predicted candidates
        self.list_pred_candidates = []
        for line in yield_lines(filepath):
            
            chunks = line.split('\t')
            complex_word = chunks[1]
            candidates = chunks[2:]
            if self.filter_complex_word:
                candidates = remove_word_from_list(complex_word, candidates) 
            candidates = unique(candidates)
            self.list_pred_candidates.append(candidates)
            
    def read_files(self, pred_filepath, gold_filepath):
        self.load_gold_data(gold_filepath)
        self.load_predicted_data(pred_filepath)
            
    def accuracy_at_1(self):
        return accuracy_at_1(self.list_pred_candidates, self.list_gold_candidates)

    def accuracy_at_k_at_top_1(self, k):
        return accuracy_at_k_at_top_gold_1(self.list_pred_candidates, self.list_sorted_gold_candidates, k)

    def precision_metrics_at_k(self, k):
        return precision_metrics_at_k(self.list_pred_candidates, self.list_gold_candidates, k)
    
    def MAP_at_K(self, k):
        return MAP_at_k(self.list_pred_candidates, self.list_gold_candidates, k)
    

    
def evaluate_file(pred_filepath, gold_filepath, results_filepath, verbose=False):
    evaluator = Evaluator()
    evaluator.read_files(pred_filepath, gold_filepath)
    results = {}
    results[f'ACC@1'] = f'{evaluator.accuracy_at_1():.4f}'
    # compute accuracy
    for k in [1, 2, 3, 4, 5]:
        results[f'ACC@{k}@Top1'] = normalize(evaluator.accuracy_at_k_at_top_1(k))
        
    # compute MAP
    for k in [1, 2, 3, 4, 5, 10]:
        results[f'MAP@{k}'] = normalize(evaluator.MAP_at_k(k))
    
    # compute potential, precision, Reecall, ... 
    tmp_result = {'potential': [], 'precision': [], 'recall': []} 
    for k in [1, 2, 3, 4, 5, 10]:
        
        values = evaluator.precision_metrics_at_k(k)
        tmp_result['potential'].append((f'Potential@{k}', values['potential']))
        tmp_result['precision'].append((f'Precision@{k}', values['precision']))
        tmp_result['recall'].append((f'Recall@{k}', values['recall']))
                                    
    for key in tmp_result:
        for metric, value in tmp_result[key]:
            results[metric] = normalize(value)
    if verbose:
        for (key, value) in results.items():
            print(f'{key:<15}: {value}')
        
    pd.DataFrame([results]).to_csv(results_filepath, index=False)

if __name__=='__main__':
    
    model_dirs = list(EXP_DIR.glob('*'))
    for model_dir in model_dirs:
        # print(model_dir)
        for dataset in [Dataset.LexMTurk, Dataset.BenchLS, Dataset.NNSeval]:
            gold_filepath = get_dataset_filepath(dataset)
            output_dir = EXP_DIR / model_dir / 'outputs/'
            filepaths = list(output_dir.glob(f'{dataset}*.tsv'))
            if len(filepaths) > 0:
                # print(filepaths[0])
                pred_filepath = filepaths[0]
                print(pred_filepath)
                results_filepath = pred_filepath.parent / f'new_eval_scores_{pred_filepath.stem}.csv'
                evaluate_file(pred_filepath, gold_filepath, results_filepath, verbose=True)
                print(f'Scores are written to file: {results_filepath}')

