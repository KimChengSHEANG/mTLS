

python train_and_eval_t5-large-TLS-3.py

n_trials=100
for lang in en;
do
    python tokens_pruning_nocand.py --n-trials=$n_trials --lang=$lang  --phase=valid

done
