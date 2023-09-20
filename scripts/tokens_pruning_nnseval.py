from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from source.constants import RESOURCES_DIR
from source.preprocessor import Preprocessor

from source.evaluate import evaluate_on, evaluation_pruning
from source.resources import EXP_DIR, Dataset, get_last_experiment_dir, split_data_train_valid_test
import optuna
import pandas as pd 
import argparse


model_name, lang, phase = None, None, None
def run_tuning(params):
    features_kwargs = {
            'CandidateRanking': {'target_ratio': 1.00},
            'WordLength': {'target_ratio': params['WordLength']},
            'WordRank': {'target_ratio': params['WordRank']},
            'WordSyllable':{'target_ratio': params['WordSyllable']},
            'SentenceSimilarity':{'target_ratio': params['SentenceSimilarity']}
        }
    print(features_kwargs)
        
    preprocessor = Preprocessor(features_kwargs, lang)
    test_set = pd.read_csv(RESOURCES_DIR / 'mask_pred_candidates/NNSeval_topk_10_roberta-base.csv')
    test_set = preprocessor.preprocess_valid_or_test_set(test_set, use_mask_pred_candidates=False)
    
    return evaluation_pruning(test_set, 'NNSeval', features_kwargs, phase=phase, lang=lang, model_dirname=model_name)

def objective(trial: optuna.trial.Trial) -> float:
    params = {
        # 'CandidateRanking': trial.suggest_float('CandidateRanking', 0.30, 1.00, step=0.05),
        'WordLength': trial.suggest_float('WordLength', 0.50, 2.00, step=0.05),
        'WordRank': trial.suggest_float('WordRank', 0.50, 2.00, step=0.05),
        'WordSyllable': trial.suggest_float('WordSyllable', 0.50, 2.00, step=0.05),
        'SentenceSimilarity': trial.suggest_float('SentenceSimilarity', 0.30, 1.00, step=0.05),
        
    }
    return run_tuning(params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokens search')
    parser.add_argument('--model-name', type=str, help='Model dir name', default=None)
    parser.add_argument('--phase', type=str, help='valid or test set', default='valid')
    parser.add_argument('--n-trials', type=int, help='Number of trials')
    parser.add_argument('--lang', type=str, help='Language: en for English, es for Spanish, pt for Portuguese')
    args = parser.parse_args()
    if args.model_name is None:
        model_name = str(get_last_experiment_dir()).split('/')[-1]
    else:
        model_name = args.model_name
    lang = args.lang
    phase = args.phase

    tuning_log_dir = Path('.') / 'tokens_pruning'
    tuning_log_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=model_name, direction="maximize",
                                storage=f'sqlite:///{tuning_log_dir}/study.db', load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials)

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")