

python train_and_eval_t5-large-TLS-2

n_trials=150
for lang in en;
do
    python tokens_pruning.py --n-trials=$n_trials --lang=$lang  --phase=valid 

done
