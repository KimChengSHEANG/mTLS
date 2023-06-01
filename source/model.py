
import pytorch_lightning as pl
from source.generate import generate
from source.metrics import accuracy_at_1, accuracy_at_k_at_top_gold_1, flatten, precision_metrics_at_k, remove_word_from_list
from source.resources import Dataset
from pytorch_lightning import seed_everything
import json 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,get_linear_schedule_with_warmup, AutoConfig,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
)

class MyModel(pl.LightningModule):
    def __init__(self, data_train, data_valid, model_name, learning_rate, adam_epsilon, weight_decay, dataset,
                 train_batch_size, valid_batch_size, max_seq_length,
                 n_gpu, gradient_accumulation_steps, num_train_epochs, warmup_steps, nb_sanity_val_steps,
                 *args, **kwargs):
        super(MyModel, self).__init__()
        self.save_hyperparameters()
        
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.model_name, device_map="auto", load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        # if 'mt5' in self.hparams.model_name:
        #     self.tokenizer.src_lang = f'{self.hparams.lang}_XX'
        #     self.tokenizer.tgt_lang = f'{self.hparams.lang}_XX'
        
        # self.model = self.model.to(self.device)

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)

    
    def training_step(self, batch, batch_idx):
        
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 
        
        loss = self(input_ids=batch["source_ids"], 
                        attention_mask=batch["source_mask"], 
                        labels=labels, 
                        decoder_attention_mask=batch['target_mask']).loss
        
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.train_batch_size)
        return loss

    
    def validation_step(self, batch, batch_idx):
        
        list_pred_candidates = []
        list_gold_candidates = []
        
        for text, complex_word, candidates in zip(batch['source'], batch['complex_word'], batch['candidates']):
            list_gold_candidates.append(json.loads(candidates))     
            pred_sents, pred_candidates = generate(text, self.model, self.tokenizer, self.hparams.max_seq_length)
            pred_candidates = remove_word_from_list(complex_word, pred_candidates) # remove pred candidates the same as complex word
            list_pred_candidates.append(pred_candidates)
            print(f'Source: {text}')
            print('Predicted sentences:')
            print('\n'.join(pred_sents))
            # print("Valid predicted sentences: ", pred_candidates)
            
            
        acc1top1 = accuracy_at_k_at_top_gold_1(list_pred_candidates, list_gold_candidates, k=1)
        acc2top1 = accuracy_at_k_at_top_gold_1(list_pred_candidates, list_gold_candidates, k=2)
        acc3top1 = accuracy_at_k_at_top_gold_1(list_pred_candidates, list_gold_candidates, k=3)
        
        list_gold_candidates = [flatten(gold_candidates) for gold_candidates in list_gold_candidates]
        # acc1 = accuracy_at_1(list_pred_candidates, list_gold_candidates)
        # val_loss = -np.exp(0.75 * np.log(acc1top1) + 0.5 * np.log(acc2top1) + 0.25 * np.log(acc3top1) + 0.75 * np.log(acc1))
        
        p1 = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k=1)['potential']
        p3 = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k=3)['potential']
        p5 = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k=5)['potential']
        p10 = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k=10)['potential']
        
        r1 = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k=1)['recall']
        r3 = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k=3)['recall']
        r5 = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k=5)['recall']
        r10 = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k=10)['recall']
        
        # m15 = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k=15)['potential']
        # val_loss = -np.exp( 0.75 * np.log(m5['potential']) + 0.5 * np.log(p10['potential'])+ 0.25 * np.log(m15['potential'])) 
        # val_loss = - p10['potential']
        
        # val_loss = -np.exp(0.4 * np.log(acc1top1) + 0.15 * np.log(acc2top1) + 0.15* np.log(acc3top1) + 0.3 * np.log(p10))
        # val_loss = -np.exp( 0.2 * np.log(m1) + 0.2 * np.log(m3) + 0.3 * np.log(m5) + 0.3 * np.log(p10)) #1
        # val_loss = -np.exp( 0.2 * np.log(acc1top1) + 0.1 * np.log(acc2top1) + 0.1 * np.log(acc3top1) + 0.1 * np.log(p1) + 0.1 * np.log(m3) + 0.2 * np.log(m5) + 0.2 * np.log(p10)) #2
        # val_loss = -np.exp( 0.4 * np.log(acc1top1) + 0.15 * np.log(p1) + 0.15 * np.log(p5) + 0.3 * np.log(p10)) #3
        # val_loss = -np.exp(0.3 * np.log(acc1top1) + 0.1 * np.log(acc2top1) + 0.1 * np.log(acc3top1) + 0.1 * np.log(p1) + 0.1 * np.log(p3) +  0.1 * np.log(p5) + 0.2 * np.log(p10)) 
        # val_loss = -np.exp( 0.5 * np.log(acc1top1) + 0.5 * np.log(p10)) #5
        # val_loss = -np.exp(np.log(acc3top1)) #6 best: en, 
        # val_loss = -np.exp(0.3 * np.log(acc1top1) + 0.15 * np.log(acc2top1)  + 0.15 * np.log(acc3top1) + 0.2 * np.log(p1) + 0.2 * np.log(p10)) #4
        # val_loss = -np.exp(np.log(p10)) #7
        # val_loss = -np.exp(np.log(p10))
        # val_loss = -np.exp( 0.5 * np.log(acc1top1) + 0.2 * np.log(acc3top1) + 0.3 * np.log(r5) ) #3
        val_loss = - np.exp(np.log(acc1top1))
        # val_loss = - np.exp(0.4 * np.log(acc1top1) + 0.2 * np.log(p1) +  0.4 * np.log(p10))
        # val_loss = - np.exp(0.4 * np.log(acc1top1) + 0.3 * np.log(acc2top1) +  0.3 * np.log(acc3top1))
        print(f'val_loss: {val_loss}') 
        self.log('val_loss', val_loss, on_step=True, prog_bar=True, logger=True, batch_size=self.hparams.valid_batch_size)
       
        return val_loss
    
    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if all(nd not in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None, using_native_amp=None, using_lbfgs=None):
        optimizer.step(closure=optimizer_closure)
        self.lr_scheduler.step()
        optimizer.zero_grad()

    def train_dataloader(self):
        
        train_dataset = TrainDataset(data=self.hparams.data_train, 
                                     tokenizer=self.tokenizer,
                                     max_len=self.hparams.max_seq_length) 
        
        dataloader = DataLoader(train_dataset,
                                batch_size=self.hparams.train_batch_size,
                                drop_last=True,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=1)
        t_total = ((len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                   * self.hparams.gradient_accumulation_steps * float(self.hparams.num_train_epochs))
        scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = EvalDataset(self.hparams.data_valid)
        return DataLoader(
            val_dataset,
            batch_size=self.hparams.valid_batch_size,
            drop_last=True,
            pin_memory=True,
            shuffle=False,
            num_workers=1,
        )
        

class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        row = self.data.iloc[index]
        # print(row)
        source_sent = row['source']
        target_sent = row['target']

        tokenized_inputs = self.tokenizer(
            [source_sent],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        tokenized_targets = self.tokenizer(
            [target_sent],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()

        src_mask = tokenized_inputs["attention_mask"].squeeze()  # might need to squeeze
        target_mask = tokenized_targets["attention_mask"].squeeze()  # might need to squeeze
        

        return {'source_ids': source_ids, 
                'source_mask': src_mask, 
                'target_ids': target_ids, 
                'target_mask': target_mask, 
                'source': source_sent, 
                'target': target_sent,
                'complex_word': row['complex_word'],
                'candidates': row['candidates']
                }


class EvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        return {'source': row['source'], 
                'complex_word': row['complex_word'], 
                'complex_word_index': row['complex_word_index'], 
                'candidates': row['candidates']}

