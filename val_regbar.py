from model_components import ClassificationModel
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader
from utilities import TrainArgs, get_model_setup
from pipeline import ClassificationPipeline
import torch
import pandas as pd

torch.manual_seed(2022)

parser = argparse.ArgumentParser(description='validation')
parser.add_argument('--model', required=True)
args = parser.parse_args()

# Model Path
model_path = get_model_setup(args)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('We will use the GPU:', torch.cuda.get_device_name(0), flush=True)
else:
    device = torch.device("cpu")

train_args = TrainArgs(total_steps=None,
                       num_epoch=5,
                       per_device_training_batch=16,
                       per_device_eval_batch=32,
                       lr=3e-5,
                       num_warmup=0,
                       log_every_n_steps=20,
                       eval_every_n_steps=None,
                       save_every_n_steps=None,
                       num_workers=0,
                       max_seq_len=None,
                       model_path=None,
                       train_data_path='data/regbar/regbar_train.csv',
                       eval_data_path='data/regbar/regbar_test.csv',
                       spacy_model_path=None,
                       tokenizer_path=None,
                       gradient_accumulation_steps=1,
                       gradient_checkpointing=False,
                       checkpoint_path=None,
                       continue_training=None,
                       dropout_prob=None)

model = ClassificationModel(model=args.model, nclass=2,  model_path=model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
print(f"Max seq length {tokenizer.model_max_length}", flush=True)

train_data = load_dataset('csv', data_files=train_args.train_data_path, cache_dir='cache/regbar/')['train']
test_data = load_dataset('csv', data_files=train_args.eval_data_path, cache_dir='cache/regbar/')['train']

train_dataloader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=train_args.per_device_training_batch,
                              pin_memory=True,
                              num_workers=train_args.num_workers)
test_dataloader = DataLoader(test_data,
                             shuffle=True,
                             batch_size=train_args.per_device_eval_batch,
                             pin_memory=True,
                             num_workers=train_args.num_workers)

# Train
model.to(device)

trainer = ClassificationPipeline(train_args,
                                 model,
                                 tokenizer,
                                 device,
                                 train_dataloader,
                                 test_dataloader,
                                 seg1_colname='text',
                                 seg2_colname=None,
                                 label_colname='label',
                                 global_attn=False)
trainer.train()
