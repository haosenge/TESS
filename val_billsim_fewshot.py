from model_components import ClassificationModel
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader
from utilities import TrainArgs, get_model_setup
import torch
import transformers
import datasets
import pandas as pd
from pipeline import ClassificationPipeline

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_warning()

torch.manual_seed(1)

parser = argparse.ArgumentParser(description='validation')
parser.add_argument('--model', required=True)
# args = parser.parse_args(['--model', 'long'])
args = parser.parse_args()


model_path = get_model_setup(args)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('We will use the GPU:', torch.cuda.get_device_name(0), flush=True)
else:
    device = torch.device("cpu")


train_args = TrainArgs(total_steps=None,
                       num_epoch = 5,
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

num_sample = [50, 200, 500, 800, 1000]

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
print(f"Max seq length {tokenizer.model_max_length}", flush=True)

acc_list = []
f1_list = []
for sample_size in num_sample:

    model = ClassificationModel(model=args.model, nclass=5, model_path=model_path)

    print(f"Sample Size {sample_size}")

    train_data = load_dataset('csv',
                              data_files=[train_args.train_data_path],
                              cache_dir="cache/billsim/")
    train_data = train_data.shuffle()['train'].select(range(sample_size))
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
                                     global_attn=False,
                                     linear_decay=True)

    val_result = trainer.train()
    best_acc = max([x['accuracy'] for x in val_result])
    best_f1 = max([x['f1'] for x in val_result])

    acc_list.append(best_acc)
    f1_list.append(best_f1)

result = {'acc':acc_list, 'f1':f1_list}
pd.DataFrame(result).to_csv(f'results/billsim_results_{args.model}.csv')
print(pd.DataFrame(result))
