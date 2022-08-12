import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_constant_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class ClassificationPipeline:

    def __init__(self,
                train_args,
                model,
                tokenizer,
                device,
                train_dataloader,
                test_dataloader,
                seg1_colname,
                seg2_colname,
                label_colname,
                global_attn=False,
                linear_decay=True):

        self.train_args=train_args
        self.model=model
        self.tokenizer=tokenizer
        self.train_dataloader=train_dataloader
        self.test_dataloader=test_dataloader
        self.seg1_colname=seg1_colname
        self.seg2_colname=seg2_colname
        self.label_colname=label_colname
        self.global_attn=global_attn
        self.device=device
        self.linear_decay=linear_decay

    def train(self):

        validation_result = []

        num_epoch = self.train_args.num_epoch
        optimizer = AdamW(self.model.parameters(),
                          lr=self.train_args.lr,
                          eps=1e-8)

        if self.linear_decay is True:
            lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                           num_warmup_steps=self.train_args.num_warmup,
                                                           num_training_steps=len(self.train_dataloader) * num_epoch)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=self.train_args.num_warmup)

        self.model.train()
        self.model.zero_grad()
        progress_bar = tqdm(range(len(self.train_dataloader) * num_epoch))

        print('================ Begin Training ==================', flush=True)
        total_loss = 0
        total_step = 0

        for ep_id in range(num_epoch):

            self.model.train()

            print(f"================= Epoch {ep_id + 1} ===============", flush=True)

            for step, batch in enumerate(self.train_dataloader):

                total_step += 1

                seg1 = batch[self.seg1_colname] if self.seg1_colname is not None else None
                seg2 = batch[self.seg2_colname] if self.seg2_colname is not None else None

                if seg1 is None:
                    token_output = self.tokenizer(seg2, truncation=True, padding=True)
                elif seg2 is None:
                    token_output = self.tokenizer(seg1, truncation=True, padding=True)
                else:
                    token_output = self.tokenizer(seg1, seg2, truncation=True, padding=True)

                model_input = {key: torch.tensor(value).to(self.device) for key, value in token_output.items()}

                if self.global_attn is True:
                    # Add global attention
                    model_input['global_attention_mask'] = torch.zeros(model_input['attention_mask'].shape,
                                                                       dtype=torch.long, device=self.device)
                    model_input['global_attention_mask'][:, 0] = 1

                labels = batch[self.label_colname].to(self.device)
                model_input['labels'] = labels

                output = self.model(**model_input)

                loss = output['loss'] / self.train_args.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)  # Clip gradient
                total_loss += loss.item() * self.train_args.gradient_accumulation_steps

                if (total_step % self.train_args.gradient_accumulation_steps) == 0 or (step == len(self.train_dataloader) - 1):
                    optimizer.step()
                    lr_scheduler.step()
                    self.model.zero_grad()

                progress_bar.update()

                if (total_step % self.train_args.log_every_n_steps == 0 or step == len(self.train_dataloader) - 1) and step != 0:
                    # print(f"Current Loss {loss.item()}", flush=True)
                    print(f'Average Loss {total_loss / total_step}', flush=True)

            # Validation
            #print(self.model.model.encoder.tess_layer_groups[0].attentions[0].query.weight)
            epoch_validation = self.validate()
            validation_result.append(epoch_validation)

        return validation_result


    def validate(self):

        print('========== Begin Validation ==============', flush=True)

        self.model.eval()
        y_pred_all = []
        y_true_all = []

        for val_step, val_batch in enumerate(self.test_dataloader):

            val_seg1 = val_batch[self.seg1_colname] if self.seg1_colname is not None else None
            val_seg2 = val_batch[self.seg2_colname] if self.seg2_colname is not None else None

            if val_seg1 is None:
                token_output = self.tokenizer(val_seg2, truncation=True, padding=True)
            elif val_seg2 is None:
                token_output = self.tokenizer(val_seg1, truncation=True, padding=True)
            else:
                token_output = self.tokenizer(val_seg1, val_seg2, truncation=True, padding=True)

            model_input = {key: torch.tensor(value).to(self.device) for key, value in token_output.items()}
            labels = val_batch[self.label_colname].to(self.device)
            model_input['labels'] = labels

            if self.global_attn is True:
                model_input['global_attention_mask'] = torch.zeros(model_input['attention_mask'].shape,
                                                                   dtype=torch.long, device=self.device)
                model_input['global_attention_mask'][:, 0] = 1

            with torch.no_grad():
                output = self.model(**model_input)

            y_pred = torch.argmax(output['prediction'], dim=1).cpu().tolist()

            y_pred_all.extend(y_pred)
            y_true_all.extend(labels.cpu().tolist())

        accuracy = accuracy_score(y_true_all, y_pred_all)
        f1 = f1_score(y_true_all, y_pred_all, average='macro')
        print(f'Accuracy is {accuracy}', flush=True)
        print(f'F1 is {f1}', flush=True)

        return {'accuracy':accuracy, 'f1':f1}




class QAPipeline:

    def __init__(self,
                train_args,
                model,
                tokenizer,
                device,
                train_dataloader,
                test_dataloader,
                seg1_colname,
                seg2_colname,
                label_colname,
                global_attn=False,
                linear_decay=True):

        self.train_args=train_args
        self.model=model
        self.tokenizer=tokenizer
        self.train_dataloader=train_dataloader
        self.test_dataloader=test_dataloader
        self.seg1_colname=seg1_colname
        self.seg2_colname=seg2_colname
        self.label_colname=label_colname
        self.global_attn=global_attn
        self.device=device
        self.linear_decay=linear_decay

    def train(self):

        validation_result = []

        num_epoch = self.train_args.num_epoch
        optimizer = AdamW(self.model.parameters(),
                          lr=self.train_args.lr,
                          eps=1e-8)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=self.train_args.num_warmup,
                                                       num_training_steps=len(self.train_dataloader) * num_epoch)
        self.model.train()
        self.model.zero_grad()
        progress_bar = tqdm(range(len(self.train_dataloader) * num_epoch))

        print('================ Begin Training ==================', flush=True)

        total_loss = 0
        total_step = 0
        max_token = self.tokenizer.model_max_length
        for ep_id in range(num_epoch):

            self.model.train()
            print(f"================= Epoch {ep_id + 1} ===============", flush=True)

            for step, batch in enumerate(self.train_dataloader):

                total_step += 1

                seg1 = batch[self.seg1_colname] if self.seg1_colname is not None else None
                seg2 = batch[self.seg2_colname] if self.seg2_colname is not None else None

                if seg1 is None:
                    token_output = self.tokenizer(seg2, truncation=True, padding=True)
                elif seg2 is None:
                    token_output = self.tokenizer(seg1, truncation=True, padding=True)
                else:
                    token_output = self.tokenizer(seg1, seg2, truncation=True, padding=True)

                model_input = {key: torch.tensor(value).to(self.device) for key, value in token_output.items()}

                if self.global_attn is True:
                    # Add global attention
                    model_input['global_attention_mask'] = torch.zeros(model_input['attention_mask'].shape,
                                                                       dtype=torch.long, device=device)
                    model_input['global_attention_mask'][:, 0] = 1

                nest_list = [x.split('|')[:max_token] for x in batch[self.label_colname]]
                label_flatten = [int(x) for sublist in nest_list for x in sublist]
                labels = torch.tensor(label_flatten).to(self.device)

                model_input['labels'] = labels
                #model_input['pad_id'] = self.tokenizer.pad_token_id
                output = self.model(**model_input)

                loss = output['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)  # Clip gradient
                total_loss += loss.item()

                optimizer.step()
                lr_scheduler.step()
                self.model.zero_grad()
                progress_bar.update()

                if (total_step % self.train_args.log_every_n_steps == 0 or step == len(self.train_dataloader) - 1) and step != 0:
                    print(f'Average Loss {total_loss / total_step}', flush=True)

            epoch_validation = self.validate()
            validation_result.append(epoch_validation)

        return validation_result


    def validate(self):

        print('========== Begin Validation ==============', flush=True)

        self.model.eval()
        y_pred_all = []
        y_true_all = []
        f1_list = []
        max_token = self.tokenizer.model_max_length
        for val_step, val_batch in enumerate(self.test_dataloader):

            val_seg1 = val_batch[self.seg1_colname] if self.seg1_colname is not None else None
            val_seg2 = val_batch[self.seg2_colname] if self.seg2_colname is not None else None

            if val_seg1 is None:
                token_output = self.tokenizer(val_seg2, truncation=True, padding=True)
            elif val_seg2 is None:
                token_output = self.tokenizer(val_seg1, truncation=True, padding=True)
            else:
                token_output = self.tokenizer(val_seg1, val_seg2, truncation=True, padding=True)

            model_input = {key: torch.tensor(value).to(self.device) for key, value in token_output.items()}
            nest_list = [x.split('|')[:max_token] for x in val_batch[self.label_colname]]  # max_token deal with truncation
            label_flatten = [int(x) for sublist in nest_list for x in sublist]
            labels = torch.tensor(label_flatten).to(self.device)
            model_input['labels'] = labels
            #model_input['pad_id'] = self.tokenizer.pad_token_id

            if self.global_attn is True:
                model_input['global_attention_mask'] = torch.zeros(model_input['attention_mask'].shape,
                                                                   dtype=torch.long, device=device)
                model_input['global_attention_mask'][:, 0] = 1

            with torch.no_grad():
                output = self.model(**model_input)


            for idx, case in enumerate(val_batch['cname']):
                gold_label = set(case.lower().split('|'))
                pred_token = model_input['input_ids'][idx][torch.sigmoid(output['prediction_padding'][idx]).view(-1) > 0.1]
                pred_word = set(self.tokenizer.decode(pred_token).lower().split())

                num_same = len(gold_label.intersection(pred_word))
                precision = 1.0 * num_same / (len(pred_word) + 1e-8)
                recall = 1.0 * num_same / len(gold_label)
                f1 = (2 * precision * recall) / (precision + recall + 1e-8)

                f1_list.append(f1)


        print(f'F1 is {np.mean(f1_list)}', flush=True)

        return {'f1':np.mean(f1_list)}
