

# python train_and_eval_mTLS.py

n_trials=100
for lang in en es pt;
do
    python tokens_pruning.py --n-trials=$n_trials --lang=$lang  --phase=valid #--model-name=exp_1679326129880862

done
