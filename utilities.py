import torch
from transformers import AlbertTokenizer
import spacy
import pickle
import gzip
import json
import glob
import os
import numpy as np
from collections import OrderedDict, Counter
from torch.distributions import Categorical
import numpy as np
# wd = "/Volumes/GoogleDrive/My Drive/Current Work/Current Projects/Pretraining"

class CollatePretrain:
    def __init__(self, tokenizer_path, max_pos):
        self.tokenizer_path = tokenizer_path
        self.max_pos = max_pos
        
    def __call__(self, batch):
        
        tokenizer = AlbertTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.model_max_length = self.max_pos
        #print(f"The tokenizer's max len is: {tokenizer.model_max_length}", flush=True)

        # Sentence Pair
        sent1 = [seq_pair['sent1'] for seq_pair in batch]
        sent2 = [seq_pair['sent2'] for seq_pair in batch]

        # print(f"Sent Pair: Sent 1: {sent1[0]} \n Sent 2: {sent2[0]}", flush=True)
        if len(sent1) != len(sent2):
            raise AssertionError('Some sequence-pair only has one element')
            
        # Shuffle 50% chance
        shuffle_indicator = torch.bernoulli(torch.tensor([0.5]*len(sent1))).bool()
        sent1_shuffle = sent1.copy()
        sent2_shuffle = sent2
        for idx in range(len(sent1)):
            if shuffle_indicator[idx] is True:
                sent1_shuffle[idx] = sent2[idx]
                sent2_shuffle[idx] = sent1[idx]
        
        sop_labels = torch.zeros(shuffle_indicator.shape)
        sop_labels[shuffle_indicator] = 1
        
        # print("sent 1: ", sent1_shuffle, flush = True)
        # print("sent 2: ", sent2_shuffle, flush = True)
        # Tokenizer
        #possible_pad_values = [512 * x for x in range(1,9)]
        #check_output = tokenizer(sent1_shuffle, sent2_shuffle, truncation=False, padding=False)
        #max_len = max(len(x) for x in check_output['input_ids'])
        #pad_value = [y for y in possible_pad_values if y > max_len][0] if max_len < 4096 else 4096
        #print(max_len, pad_value, flush=True)
        #output = tokenizer(sent1_shuffle, sent2_shuffle, truncation = True, padding = 'max_length', max_length = pad_value)
        output = tokenizer(sent1_shuffle, sent2_shuffle, truncation=True, padding=True)
        output = {key:torch.tensor(value) for key, value in output.items()}
        # print(output['input_ids'].shape, flush=True)
        # Mask Language Modeling
        mask_input, mlm_labels = self._masking_tokens(output, tokenizer.mask_token_id, 
                                                  tokenizer.bos_token_id, tokenizer.eos_token_id, 0.15)

        # Deal with Attention Mask
        output['global_attention_mask'] = torch.zeros(output['attention_mask'].shape, dtype=torch.long)
        output['global_attention_mask'][:, 0] = 1

        # mod_mask = output['attention_mask'].clone()
        # # mod_mask[mod_mask ==0] = -10000
        # # mod_mask[mod_mask ==1] = 0
        # mod_mask[:,0] = 2 # Global attention on CLS
        # output['attention_mask'] = mod_mask
        
        return {'input_ids':mask_input, 
                'token_type_ids':output['token_type_ids'], 
                'attention_mask':output['attention_mask'],
                'global_attention_mask': output['global_attention_mask'],
                'labels': mlm_labels,
                'sentence_order_label': sop_labels}
    
    @staticmethod
    def _masking_tokens(input_dict, mask_token_id,  bos_token_id, eos_token_id, masking_prob):
        
        if not ((masking_prob <= 1) & (masking_prob >= 0)): raise AssertionError
        if not torch.sum(input_dict['input_ids'] == mask_token_id) == 0: raise AssertionError
        
        feature = input_dict['input_ids']
        attn_mask = input_dict['attention_mask']

        # masking_prob coin for each entry
        prob_tensor = torch.empty(feature.shape).fill_(masking_prob)
        mask_torch = torch.bernoulli(prob_tensor).bool()

        mask_torch[:,0] = False  # Dont mask CLS
        mask_torch[attn_mask == 0] = False # Dont mask Padding
        mask_torch[feature == eos_token_id] = False # Dont mask SEP

        # if there is no mask, randomly add one mask
        assert feature.shape == attn_mask.shape
        if torch.sum(mask_torch) == 0:
            valid_positions = torch.logical_and(torch.logical_and(feature.view(-1) != bos_token_id, feature.view(-1) != eos_token_id), attn_mask.view(-1) != 0)
            valid_id = torch.nonzero(valid_positions.float(), as_tuple = False)
            random_idx = np.random.choice(valid_id.reshape(-1), size = 1)[0]
            # Mask
            mask_torch.view(-1)[random_idx] = True

        mask_num = torch.sum(mask_torch)
        assert mask_num != 0

        # Label
        labels = torch.empty(feature.shape).fill_(-100).long()
        labels[mask_torch] = feature[mask_torch]  # True answer and padding

        # Masked input
        masked_input = feature.clone()
        masked_input[mask_torch] = mask_token_id
        
        if not torch.sum(masked_input == mask_token_id) == mask_num: raise AssertionError
        if not torch.sum(labels != -100) == mask_num: raise AssertionError
        
        return masked_input, labels


class CollateMLM:
    def __init__(self, tokenizer_path, max_pos):
        self.tokenizer_path = tokenizer_path
        self.max_pos = max_pos

    def __call__(self, batch):

        tokenizer = AlbertTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.model_max_length = self.max_pos

        # Sentence Pair
        sent1 = [seq_pair['sent1'] for seq_pair in batch]
        sent2 = [seq_pair['sent2'] for seq_pair in batch]
        output = tokenizer(sent1, sent2, truncation=True, padding=True)
        output = {key: torch.tensor(value) for key, value in output.items()}

        # Mask Language Modeling
        mask_input, mlm_labels = self._masking_tokens(output, tokenizer.mask_token_id,
                                                      tokenizer.bos_token_id, tokenizer.eos_token_id, 0.15)

        # Deal with Attention Mask
        output['global_attention_mask'] = torch.zeros(output['attention_mask'].shape, dtype=torch.long)
        output['global_attention_mask'][:, 0] = 1


        return {'input_ids': mask_input,
                'token_type_ids': output['token_type_ids'],
                'attention_mask': output['attention_mask'],
                'global_attention_mask': output['global_attention_mask'],
                'labels': mlm_labels}

    @staticmethod
    def _masking_tokens(input_dict, mask_token_id, bos_token_id, eos_token_id, masking_prob):

        if not ((masking_prob <= 1) & (masking_prob >= 0)): raise AssertionError
        if not torch.sum(input_dict['input_ids'] == mask_token_id) == 0: raise AssertionError

        feature = input_dict['input_ids']
        attn_mask = input_dict['attention_mask']

        # masking_prob coin for each entry
        prob_tensor = torch.empty(feature.shape).fill_(masking_prob)
        token_mask = torch.bernoulli(prob_tensor).bool()

        # N-gram
        N = attn_mask.sum(axis=1)
        deno = np.log(N).view(1,-1)  # log approximation
        p = torch.cat([1/deno, 2/deno, 3/deno],0)
        sampler = Categorical(probs=p)

        mask_torch = token_mask.clone()
        for sample_id in range(token_mask.shape[0]):
            for token_id in range(token_mask.shape[1]):
                if token_mask[sample_id, token_id] is True:
                    # Check Span
                    span_len = sampler.sample()[sample_id]
                    mask_torch[sample_id, token_id:token_id+span_len+1] = True

        mask_torch[:, 0] = False  # Dont mask CLS
        mask_torch[attn_mask == 0] = False  # Dont mask Padding
        mask_torch[feature == eos_token_id] = False  # Dont mask SEP

        # if there is no mask, randomly add one mask
        assert feature.shape == attn_mask.shape
        if torch.sum(mask_torch) == 0:
            valid_positions = torch.logical_and(
                torch.logical_and(feature.view(-1) != bos_token_id, feature.view(-1) != eos_token_id),
                attn_mask.view(-1) != 0)
            valid_id = torch.nonzero(valid_positions.float(), as_tuple=False)
            random_idx = np.random.choice(valid_id.reshape(-1), size=1)[0]
            # Mask
            mask_torch.view(-1)[random_idx] = True

        mask_num = torch.sum(mask_torch)
        assert mask_num != 0

        # Label
        labels = torch.empty(feature.shape).fill_(-100).long()
        labels[mask_torch] = feature[mask_torch]  # True answer and padding

        # Masked input
        masked_input = feature.clone()
        masked_input[mask_torch] = mask_token_id

        if not torch.sum(masked_input == mask_token_id) == mask_num: raise AssertionError
        if not torch.sum(labels != -100) == mask_num: raise AssertionError

        return masked_input, labels


class AlbertCollateMLM:
    def __init__(self, tokenizer_path, max_pos):
        self.tokenizer_path = tokenizer_path
        self.max_pos = max_pos

    def __call__(self, batch):

        tokenizer = AlbertTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.model_max_length = self.max_pos

        # Sentence Pair
        sent1 = [seq_pair['sent1'] for seq_pair in batch]
        sent2 = [seq_pair['sent2'] for seq_pair in batch]
        output = tokenizer(sent1, sent2, truncation=True, padding=True)
        output = {key: torch.tensor(value) for key, value in output.items()}

        # Mask Language Modeling
        mask_input, mlm_labels = self._masking_tokens(output, tokenizer.mask_token_id,
                                                      tokenizer.bos_token_id, tokenizer.eos_token_id, 0.15)



        return {'input_ids': mask_input,
                'token_type_ids': output['token_type_ids'],
                'attention_mask': output['attention_mask'],
                'labels': mlm_labels}

    @staticmethod
    def _masking_tokens(input_dict, mask_token_id, bos_token_id, eos_token_id, masking_prob):

        if not ((masking_prob <= 1) & (masking_prob >= 0)): raise AssertionError
        if not torch.sum(input_dict['input_ids'] == mask_token_id) == 0: raise AssertionError

        feature = input_dict['input_ids']
        attn_mask = input_dict['attention_mask']

        # masking_prob coin for each entry
        prob_tensor = torch.empty(feature.shape).fill_(masking_prob)
        token_mask = torch.bernoulli(prob_tensor).bool()

        # N-gram
        N = attn_mask.sum(axis=1)
        deno = np.log(N).view(1,-1)  # log approximation
        p = torch.cat([1/deno, 2/deno, 3/deno],0)
        sampler = Categorical(probs=p)

        mask_torch = token_mask.clone()
        for sample_id in range(token_mask.shape[0]):
            for token_id in range(token_mask.shape[1]):
                if token_mask[sample_id, token_id] is True:
                    # Check Span
                    span_len = sampler.sample()[sample_id]
                    mask_torch[sample_id, token_id:token_id+span_len+1] = True

        mask_torch[:, 0] = False  # Dont mask CLS
        mask_torch[attn_mask == 0] = False  # Dont mask Padding
        mask_torch[feature == eos_token_id] = False  # Dont mask SEP

        # if there is no mask, randomly add one mask
        assert feature.shape == attn_mask.shape
        if torch.sum(mask_torch) == 0:
            valid_positions = torch.logical_and(
                torch.logical_and(feature.view(-1) != bos_token_id, feature.view(-1) != eos_token_id),
                attn_mask.view(-1) != 0)
            valid_id = torch.nonzero(valid_positions.float(), as_tuple=False)
            random_idx = np.random.choice(valid_id.reshape(-1), size=1)[0]
            # Mask
            mask_torch.view(-1)[random_idx] = True

        mask_num = torch.sum(mask_torch)
        assert mask_num != 0

        # Label
        labels = torch.empty(feature.shape).fill_(-100).long()
        labels[mask_torch] = feature[mask_torch]  # True answer and padding

        # Masked input
        masked_input = feature.clone()
        masked_input[mask_torch] = mask_token_id

        if not torch.sum(masked_input == mask_token_id) == mask_num: raise AssertionError
        if not torch.sum(labels != -100) == mask_num: raise AssertionError

        return masked_input, labels


class CollateValidation:
    
    def __init__(self, tokenizer_path, max_pos = 4096):
        
        self.tokenizer_path = tokenizer_path
        self.max_pos = max_pos
    
    def __call__(self, batch):
        
        tokenizer = AlbertTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.model_max_length = self.max_pos
        
        if 'text' in batch[0].index: # determine sentence-pair or not
            feature_batch = [x.text for x in batch]
            output = tokenizer(feature_batch, truncation = True, padding = True)

        else:
            sent1_batch = [x.sentence1 for x in batch]
            sent2_batch = [x.sentence2 for x in batch]
            output = tokenizer(sent1_batch, sent2_batch, truncation = True, padding = True)
        
        output = {key:torch.tensor(value) for key, value in output.items()}
        labels = torch.tensor([x.label for x in batch])
        
        return {'input_ids':output['input_ids'], 
                'attention_mask':output['attention_mask'],
                'token_type_ids':output['token_type_ids'],
                'labels': labels}
    
    
    
class TrainArgs:
    
    def __init__(self,
                 model_path,
                 train_data_path,
                 eval_data_path,
                 spacy_model_path,
                 tokenizer_path,
                 total_steps=1000,
                 num_epoch=None,
                 per_device_training_batch=1,
                 per_device_eval_batch=1,
                 lr = 3e-5,
                 num_warmup = 100,
                 log_every_n_steps = 10,
                 num_workers = 0,
                 max_seq_len = 512,
                 eval_every_n_steps = 1000,
                 save_every_n_steps = 10000,
                 gradient_accumulation_steps = 10,
                 gradient_checkpointing = False,
                 seed=1,
                 checkpoint_path='model/base_checkpoint.pt',
                 continue_training=False,
                 dropout_prob = None):

        self.model_path = model_path
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.spacy_model_path = spacy_model_path
        self.tokenizer_path = tokenizer_path
        self.total_steps = total_steps
        self.per_device_training_batch = per_device_training_batch
        self.per_device_eval_batch = per_device_eval_batch
        self.lr = lr
        self.num_warmup = num_warmup
        self.log_every_n_steps = log_every_n_steps
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.eval_every_n_steps = eval_every_n_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.seed = seed
        self.checkpoint_path=checkpoint_path
        self.save_every_n_steps = save_every_n_steps
        self.continue_training = continue_training
        self.num_epoch = num_epoch if self.total_steps is None else ValueError('Cannot set num_epoch and total_steps at the same time')
        self.dropout_prob = dropout_prob
    

class SegmentData:
    
    def __init__(self, tokenizer_path, spacy_model_path, data_path):
        
        self.tokenizer_path = tokenizer_path
        self.spacy_model_path = spacy_model_path
        self.data_path = data_path

    def sentence_split(self, lines_per_load, max_length, save_to_file = False):
        
        all_chunks = []
        current_chunk = ''
        line_count = 0
        
        with open(self.data_path, 'r', encoding='utf-8') as file:

            for line in file: # Iterate through files
                
                line_clean = line.strip()
                if line_clean == '':  # Don't inlucde empty line
                    continue
            
                if line_count < lines_per_load:
                    
                    current_chunk += ' ' + line_clean
                    line_count += 1
                    
                elif line_count >= lines_per_load:
                    
                    splited_chunck = self._split(current_chunk, max_length)
                    
                    # The last sentence may be incomplete, push it to the next chunk
                    all_chunks.extend(splited_chunck[:-2])
                    current_chunk = splited_chunck[-1]
                    line_count = 0
                    
        # Save the last chunk
        splited_chunck = self._split(current_chunk, max_length)
        all_chunks.extend(splited_chunck)
            
        if save_to_file is not False:
            
            if isinstance(save_to_file, str) == False:
                raise ValueError('save_to_file should be the filename')
                
            result = {'data_list': all_chunks}
            with open(save_to_file, 'wb') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return all_chunks
        
    def _split(self, text, max_length):
        
        tokenizer = AlbertTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.model_max_length = max_length
        
        nlp = spacy.load(self.spacy_model_path, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
        nlp.enable_pipe("senter")
        nlp.max_length = len(text)
        
        text_doc = nlp(text)
        length = 0
        tok_sents = ""
        tok_sents_list = []
    
        for i, sent in enumerate(text_doc.sents):
            
            sent_length = len(tokenizer.tokenize(sent.text))
            
            if sent_length <= max_length:
                tok_length = len(tokenizer.tokenize(sent.text)) + length 
                if tok_length  <= max_length: 
                    tok_sents += sent.text + " "  # add sentence text to tok_sents  
                    length = tok_length 
                    if i == len(list(text_doc.sents)) - 1: 
                        tok_sents_list.append(tok_sents) # last sentence
                else: 
                    tok_sents_list.append((tok_sents))
                    # tok_sents_list.append(tokenizer.tokenize(tok_sents))
                    length = 0 
                    tok_sents = ""
                    tok_sents += sent.text + " "
                    length = len(tokenizer.tokenize(sent.text))
                    
            else:
                
                sent_list = tokenizer.tokenize(sent.text)
                while len(sent_list) > max_length:
                    
                    tok_sents = sent_list[:max_length]
                    # tok_sents_list.append(tok_sents) 
                    tok_sents_list.append(tokenizer.convert_tokens_to_string(tok_sents))
                    del sent_list[:max_length]
                    
                tok_sents = sent_list     
                # tok_sents_list.append(tok_sents)     
                
                tok_sents_list.append(tokenizer.convert_tokens_to_string(tok_sents))
                length = 0 
                tok_sents = ""
                
        return tok_sents_list      


class PretrainDataModel(torch.utils.data.Dataset):
    def __init__(self, data_path_name):
        with gzip.open(data_path_name, 'r') as fin:
            self.data_dict = json.loads(fin.read().decode('utf-8'))
        self.sent_pair = self.data_dict['data_list']

    def __len__(self):
        # return len(self.data['data_list'])
        return len(self.sent_pair)

    def __getitem__(self, idx):
        return self.sent_pair[idx]


class ValidationDataModel(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.iloc[idx]


class AnnualReportsText:
    def __init__(self, inputs_path, output_path=None, recursive =False):
        self.inputs_path = inputs_path
        self.recursive = recursive
        self.output_path = output_path


    # HG datasets format
    def reformat(self):
        for filename in glob.glob(self.inputs_path + '/' + '*.gz', recursive=True):
            data_dir, ext = os.path.splitext(filename)
            data_name = data_dir.split('/')[-1]

            with gzip.open(filename, "r") as f:
                data_dict = json.loads(f.read().decode('utf-8'))

            data = data_dict['data_list']

            [doc.append('End of Document') for doc in data]  # append method is inplace

            data = list(map(' '.join, data))

            data = [x for x in data if x]

            data_dict = [{"text": x} for x in data]

            output_filename = os.path.join(self.output_path, data_name) if self.output_path else data_dir

            with open(output_filename, "w") as file:
                for item in data_dict:
                    file.write(json.dumps(item) + "\n")


def load_cp_state_dict(model, checkpoint_file, device, **kwargs):
    """Convert DDP state dict to normal state dict"""
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    new_model_dict = OrderedDict()
    # print(ddp_model_state_dict, flush=True)
    for key, value in model_state_dict.items():

        if key.split('.')[0] == 'module':  # if ddp model
            if hasattr(model, 'albert') or hasattr(model, 'tess'):
                new_key = '.'.join(key.split('.')[1:]) # remove module.
                new_model_dict[new_key] = value
            else:
                ks = key.split('.')[1:]  # rm module
                ks[0] = 'model'  # replace albert with model
                new_key = '.'.join(ks)
                new_model_dict[new_key] = value
        else:
            if hasattr(model, 'albert'):
                new_key = key
                new_model_dict[new_key] = value
            else:
                ks = key.split('.')  # rm module
                ks[0] = 'model'  # repace albert with model
                new_key = '.'.join(ks)
                new_model_dict[new_key] = value

        # print(f"Old Key: {key}\n New Key: {new_key}", flush=True)

    print(model.load_state_dict(new_model_dict,**kwargs), flush=True)
    model.load_state_dict(new_model_dict, **kwargs)
    return model

def compute_qa_f1(token_gold, token_pred):

    common = Counter(token_gold) & Counter(token_pred)
    num_same = sum(common.values())

    if len(token_gold) == 0 or len(token_pred) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(token_gold == token_pred)
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(token_pred)
    recall = 1.0 * num_same / len(token_gold)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def get_model_setup(args):

    if args.model == 'bert':
        model_path = "model/bert_base"
    elif args.model == 'albert':
        model_path = 'model/albert_base_v2'
    elif args.model == 'long':
        model_path = 'model/'
    elif args.model == 'roberta':
        model_path = "model/roberta_base"
    elif args.model == 'tess':
        model_path = "model/tess_768"
    else:
        raise ValueError('Model name not found')

    print(f'Validate {model_path}')

    return model_path
