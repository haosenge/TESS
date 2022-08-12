from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader
from utilities import TrainArgs
import torch
from pipeline import ClassificationPipeline
import transformers
from model_components import ClassificationModel
from utilities import get_model_setup

transformers.logging.set_verbosity_error()
torch.manual_seed(1)


parser = argparse.ArgumentParser(description='validation')
parser.add_argument('--model', required=True)
# args = parser.parse_args(['--model', 'tess'])
args = parser.parse_args()

# Model Path
model_path = get_model_setup(args)

# GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('We will use the GPU:', torch.cuda.get_device_name(0), flush=True)
else:
    device = torch.device("cpu")


# Parameters
train_args = TrainArgs(total_steps=None,
                       num_epoch=3,
                       per_device_training_batch=15,
                       per_device_eval_batch=32,
                       lr=3e-5,
                       num_warmup=0,
                       log_every_n_steps=20,
                       eval_every_n_steps=None,
                       save_every_n_steps=None,
                       num_workers=0,
                       max_seq_len=None,
                       model_path=None,
                       train_data_path='data/bill_similarity/train_sub_sec_pairs.csv',
                       eval_data_path='data/bill_similarity/test_sub_sec_pairs.csv',
                       spacy_model_path=None,
                       tokenizer_path=None,
                       gradient_accumulation_steps=1,
                       gradient_checkpointing=False,
                       checkpoint_path=None,
                       continue_training=None,
                       dropout_prob=None)

# Setup Model

model = ClassificationModel(model=args.model, nclass=5, model_path=model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
print(f"Max seq length {tokenizer.model_max_length}", flush=True)


train_data = load_dataset('csv',
                          data_files=[train_args.train_data_path],
                          cache_dir="cache/billsim/")['train']
test_data = load_dataset('csv', data_files=train_args.eval_data_path, cache_dir='cache/billsim/')['train']


del_col = ['sec_a_id', 'sec_b_id', 'sec_a_title', 'sec_b_title']
train_data = train_data.remove_columns(del_col)
test_data = test_data.remove_columns(del_col)

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

# Validation
model.to(device)

trainer = ClassificationPipeline(train_args,
                                   model,
                                   tokenizer,
                                   device,
                                   train_dataloader,
                                   test_dataloader,
                                   seg1_colname='sec_a_text',
                                   seg2_colname='sec_b_text',
                                   label_colname='label',
                                   global_attn=False)
trainer.train()