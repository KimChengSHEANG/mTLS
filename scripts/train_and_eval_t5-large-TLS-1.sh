

python train_and_eval_t5-large-TLS-1.py

n_trials=50
lang=en

python tokens_pruning_mturk.py --n-trials=$n_trials --lang=$lang  --phase=valid
python tokens_pruning_nnseval.py --n-trials=$n_trials --lang=$lang  --phase=valid
python tokens_pruning_benchls.py --n-trials=$n_trials --lang=$lang  --phase=valid