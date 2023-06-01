import gc
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # fix path

import pytorch_lightning as pl
import argparse
from transformers import set_seed
from pytorch_lightning import seed_everything
from source import wandb_config
from source.helper import log_params, log_stdout, get_experiment_dir, get_device
from source.model import MyModel
import logging
import wandb, os
import torch 
import gc
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def train(data_train, data_valid, kwargs):
    args = argparse.Namespace(**kwargs)

    seed_everything(args.seed)
    set_seed(args.seed)

    print(kwargs)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="checkpoint-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        verbose=True,
        mode="min",
        save_top_k= args.save_top_k, # 3
    )
    callbacks = [checkpoint_callback]
    
    if args.early_stop_callback:
        early_stop = EarlyStopping(monitor="val_loss", patience=args.earlystop_patience, mode="min")
        callbacks.append(early_stop)
        
    wandb_logger = WandbLogger(wandb_config.WANDB_PROJECT_NAME)


    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        # accelerator='auto',
        # devices='auto',
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp_16 else 32,
        amp_backend='apex',
        amp_level=args.opt_level,
        # precision='bf16',
        # amp_backend='native',
        # gradient_clip_val=args.max_grad_norm,
        callbacks=callbacks,
        num_sanity_val_steps=args.nb_sanity_val_steps,  # skip sanity check to save time for debugging purpose
        logger=wandb_logger,
    )

    print("Initialize model")
    kwargs['data_train'] = data_train
    kwargs['data_valid'] = data_valid
    model = MyModel(**kwargs)
    trainer = pl.Trainer(**train_params)
    print(" Training model")
    trainer.fit(model)

    print("training finished")

    print('Best model path:', checkpoint_callback.best_model_path)  
    print('Best model score:', checkpoint_callback.best_model_score.item()) 
    
    # save best checkpoint path
    log_params(kwargs['output_dir'] / 'best_model.json', 
               {'best_model_path': checkpoint_callback.best_model_path,
                'best_model_score': checkpoint_callback.best_model_score.item()})


def train_on(data_train, data_valid, kwargs):
    # return
    log_params(kwargs["output_dir"] / "params.json", kwargs)
    
    os.environ['WANDB_API_KEY'] = wandb_config.WANDB_API_KEY
    os.environ['WANDB_MODE'] = wandb_config.WANDB_MODE
    wandb.init(project=wandb_config.WANDB_PROJECT_NAME, name=f"{kwargs['output_dir'].stem}_Train", job_type='Train', config=kwargs)
    

    with log_stdout(kwargs['output_dir'] / "logs.txt"):
        # torch.backends.cudnn.benchmark = True
        train(data_train, data_valid, kwargs)
    
    wandb.finish()
    
    # clear memory
    if get_device() == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    


