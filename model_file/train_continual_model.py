import os,random,time,argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AdamW,AutoTokenizer
from transformers import XLMRobertaTokenizer,XLMRobertaConfig,XLMRobertaForMaskedLM
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")# remove UndefinedMetricWarning: Precision is ill-defined


def main(args):
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model_name)
    print("[STEP 1] load XLMRobera tokenizer")
    model = XLMRobertaForMaskedLM.from_pretrained(args.pretrained_model_name)
    print('[STEP 2] load XLMRobertaForMaskedLM')
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.data_path,
        block_size=250,
    )
    print('[STEP 3] load data')
    data_collator = DataCollatorForLanguageModeling(
       tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    print('[STEP 4] load data collator for language modeling')
    training_args = TrainingArguments(
        output_dir=f"./checkpoint/{args.pretrained_model_name}-retrained",
        overwrite_output_dir=True,
        num_train_epochs=15,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        warmup_steps=1000,
        # save_steps=500,
        save_strategy='epoch',
        save_total_limit=2,
        seed=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )
    trainer.train()
    print('[STEP 5] pretraining language modeling completed!!!')

    trainer.save_model(f"./checkpoint/{args.pretrained_model_name}-retrained")
    tokenizer.save_pretrained(f"./checkpoint/{args.pretrained_model_name}-retrained")
    print('[STEP 6] language modeling saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', default='xlm-roberta-large')
    parser.add_argument('--data_path', default='/home/TRIM-AI/dataset_continual_pretraining.txt')
    parser.add_argument('--batch_size', type=int, default=16)# 16 for A100; 4 for v100
	
    args = parser.parse_args()
    main(args)