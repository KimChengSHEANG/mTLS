# Multilingual Controllable Lexical Simplification



## Requirements

**Step1**. Install PyTorch following this link: https://pytorch.org/get-started/locally/

```bash
Examples:
# gpu version
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# cpu version
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
######
```

**Step2.** Install all requirements

```bash
pip install -r requirements.txt
```



## Train and Evaluate the TLS-1 Model 

### Train

-   cd to the folder scripts

```bash
python train_and_eval_t5-large-TLS-1.py
```



### Tokens value search

```bash
python tokens_pruning_mturk.py --n-trials=50 --lang=en  --phase=valid  --model-name=None
python tokens_pruning_nnseval.py --n-trials=50 --lang=en  --phase=valid --model-name=None
python tokens_pruning_benchls.py --n-trials=50 --lang=en  --phase=valid --model-name=None

#E.g., --model-name=exp_1679326129880862  to load exp_1679326129880862 model
```



### Evaluate

-   First, update the `evaluate-TLS-1.py` file, and set the `model_dir=None` means that the script will load the latest model or set a model folder to load the specific one like `model_dir=exp_1679619309663310`  to load  `exp_1679619309663310` model.

-   Update the `features_kwargs` to the best set from the tokens search

-   And run the following script to evaluate

```bash
python evaluate-TLS-1.py
```







## Train and Evaluate the TLS-2 Model 

```bash
# Train 
python train_and_eval_t5-large-TLS-2.py
```



### Tokens value search

```bash
python tokens_pruning.py --n-trials=150 --lang=en  --phase=valid --model-name=None
#E.g., --model-name=exp_1679326129880862  to load exp_1679326129880862 model
```



### Evaluate

-   First, update the `evaluate.py` file, and set the `model_dir=None` means that the script will load the latest model or set a model folder to load the specific one like `model_dir=exp_1679619309663310`  to load  `exp_1679619309663310` model.

-   Update the `features_kwargs` to the best set from the tokens search

-   And run the following script to evaluate

```bash
python evaluate.py
```







## Train and Evaluate the TLS-3 Model 

```bash
# Train 
python train_and_eval_t5-large-TLS-3.py
```



### Tokens value search

```bash
python tokens_pruning.py --n-trials=150 --lang=en  --phase=valid --model-name=None
#E.g., --model-name=exp_1679326129880862  to load exp_1679326129880862 model
```



### Evaluate

-   First, update the `evaluate.py` file, and set the `model_dir=None` means that the script will load the latest model or set a model folder to load the specific one like `model_dir=exp_1679619309663310`  to load  `exp_1679619309663310` model.

-   Update the `features_kwargs` to the best set from the tokens search

-   And run the following script to evaluate

```bash
python evaluate.py
```







## Train and Evaluate the mTLS Model 

```bash
# Train 
python train_and_eval_mTLS.py
```



### Tokens value search

```bash
sh train_and_eval_mTLS-tokens-search.sh
```



### Evaluate

-   First, update the `evaluate.py` file, and set the `model_dir=None` means that the script will load the latest model or set a model folder to load the specific one like `model_dir=exp_1679619309663310`  to load  `exp_1679619309663310` model.

-   Update the `features_kwargs` to the best set from the tokens search for each language

-   And run the following script to evaluate

```bash
python evaluate-mTLS.py
```



