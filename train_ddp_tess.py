from transformers import AutoConfig, AlbertForMaskedLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.multiprocessing as mp
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import torch
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utilities import TrainArgs, CollatePretrain, AlbertCollateMLM, load_cp_state_dict
from model_components import TESSForMaskedLM, TESSModel
from pynvml import *
from datasets import load_dataset
import os
import time
# from tqdm.auto import tqdm

logger = logging.getLogger('monitor')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('train_logs.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

import transformers
transformers.logging.set_verbosity_error()

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


torch.manual_seed(1)

# =========== Train =================#
traindata_list = []


train_args = TrainArgs(total_steps=1000000,
                       per_device_training_batch=25,
                       per_device_eval_batch=10,
                       lr=1e-4,
                       num_warmup=1000,
                       log_every_n_steps=20,
                       eval_every_n_steps=100,
                       save_every_n_steps=2500,
                       num_workers=4,
                       max_seq_len=768,
                       model_path="model/tess_base_768_shared",
                       train_data_path=traindata_list,
                       eval_data_path='data/wikitext-103-raw/wikitext_test_segment.pickle',
                       spacy_model_path='model/en_core_web_sm-3.2.0',
                       tokenizer_path="model/albert_base_v2",
                       gradient_accumulation_steps=20,
                       gradient_checkpointing=False,
                       checkpoint_path='model/tess_checkpoint.pt',
                       continue_training=True)



def main(rank, world_size):
    # =========== Configuration ================= #
    # Model Configuration
    config = AutoConfig.from_pretrained(train_args.model_path, local_files_only=True)
    config.gradient_checkpointing = train_args.gradient_checkpointing

    # Initialize
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # ================== Init Model =========================
    model = TESSForMaskedLM.from_pretrained(train_args.model_path, local_files_only=True)
    base_model = TESSModel.from_pretrained(train_args.model_path, local_files_only=True, add_pooling_layer=False)
    model.tess = base_model
    #model.tess = model.tess.from_pretrained(train_args.model_path, local_files_only=True)

    collate_fn = AlbertCollateMLM(train_args.tokenizer_path, train_args.max_seq_len)
    # model.config.gradient_checkpointing = train_args.gradient_checkpointing
    print(f"Pre-trained Parameter {model.tess.encoder.tess_layer_groups[0].attentions[0].query.weight}", flush=True)

    if train_args.continue_training is True:
        print("Loading training checkpoints", flush=True)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(train_args.checkpoint_path, map_location=map_location)
        print(f"Last Training Data: {checkpoint['last_file']}", flush=True)
        load_cp_state_dict(model, train_args.checkpoint_path, device=map_location, strict=False)
        print(
            f"Weight after loading {model.tess.encoder.tess_layer_groups[0].attentions[0].query.weight}",
            flush=True)

    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    optimizer = AdamW(ddp_model.parameters(),
                      lr=train_args.lr,
                      eps=1e-8)


    total_step = 0
    total_train_loss = []
    if train_args.continue_training is True:
        total_step = checkpoint['total_step']


    lr_scheduler = get_constant_schedule_with_warmup(optimizer,
                                                     num_warmup_steps=train_args.num_warmup)


    # =========== Prepare to run ==============#
    ddp_model.zero_grad()
    ddp_model.train()

    if rank == 0:
        print("GPU Utilization before Training", flush=True)
        print_gpu_utilization()
    dist.barrier()

    # =========== Loop over training data list =================#
    t0 = time.time()
    for dataset in ['1']:
        #train_args.train_data_path = 'data/' + dataset
        # print(f'Training on the {dataset}', flush=True)
        logger.info(f'Training on the {dataset}')

        train_data = load_dataset('csv', data_files=train_args.train_data_path, index_col=[0], cache_dir='cache/mix/').shuffle(seed=42)['train']
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
                                                                        num_replicas=world_size,
                                                                        rank=rank,
                                                                        shuffle=True)
        train_dataloader = torch.utils.data.DataLoader(train_data,
                                                       shuffle=False,
                                                       batch_size=train_args.per_device_training_batch,
                                                       sampler=train_sampler,
                                                       num_workers=train_args.num_workers,
                                                       pin_memory=True,
                                                       collate_fn=collate_fn)

        for step, batch in enumerate(train_dataloader):

            total_step += 1

            batch = {key: value.to(rank) for key, value in batch.items()}

            output = ddp_model(**batch)

            # Gradient
            loss = output['loss'] / train_args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 5)  # Clip gradient


            if total_step % train_args.log_every_n_steps == 0 and total_step != 0:
                print_gpu_utilization()
                loss_list = [torch.zeros(1, dtype=torch.float32, device='cuda') for _ in range(world_size)]
                # Gather loss across gpus
                dist.all_gather(loss_list, loss)
                if rank == 0:  # Print only once
                    total_train_loss = total_train_loss[-100:]
                    total_train_loss.extend(loss_list[0])
                    # print(f'Current Loss: {loss.item() * train_args.gradient_accumulation_steps}', flush=True)
                    logger.info(f'Current Loss: {loss.item() * train_args.gradient_accumulation_steps}')
                    # print(f"Check Weight:{ddp_model.module.albert.encoder.layer_group_wrapper.albert_layer_groups[0].albert_layers[0].attention.longformer_self_attn.query.weight[0,1]}",flush=True)
                    # print(f"Average Training Loss at Step {total_step}: {(sum(total_train_loss)*train_args.gradient_accumulation_steps)/len(total_train_loss)}")
                    logger.info(f"Average Training Loss at Step {total_step}: {(sum(total_train_loss)*train_args.gradient_accumulation_steps)/len(total_train_loss)}")
                dist.barrier()

            if total_step % train_args.gradient_accumulation_steps == 0:
                assert ddp_model.module.tess.encoder.tess_layer_groups[0].attentions[0].query.weight.grad is not None
                # assert ddp_model.module.tess.encoder.tess_layer_groups[0].attentions[0].query.weight.grad[0,23] != \
                #       ddp_model.module.tess.encoder.tess_layer_groups[-1].attentions[0].query.weight.grad[0,23]
                optimizer.step()
                lr_scheduler.step()
                ddp_model.zero_grad()


            if total_step % train_args.save_every_n_steps == 0 and total_step != 0:  # Save every 10K steps
                if rank == 0:
                    print(f"Checkpoint Reached at step {total_step}. Save Model ...", flush=True)
                    logger.info(f"Checkpoint Reached at step {total_step}. Save Model ...")

                    torch.save({
                        "last_file": train_args.train_data_path,
                        "model_state_dict": ddp_model.state_dict(),
                        #'optimizer_state_dict': optimizer.state_dict(),
                        'total_step': total_step,
                        'total_train_loss': total_train_loss},
                        train_args.checkpoint_path)
                dist.barrier()
                print("Empty Cache ...", flush=True)
                torch.cuda.empty_cache()


            # print("Empty Cache ...", flush=True)
            torch.cuda.empty_cache()

            if total_step >= train_args.total_steps:  # Break loop over training instances
                break
        if total_step >= train_args.total_steps:  # Break loop over dataset
            break

    # Final Save
    if rank == 0:
        logger.info('Training Ends. Save Model ...')
        torch.save({
            "last_file": train_args.train_data_path,
            "model_state_dict": ddp_model.state_dict(),
            #'optimizer_state_dict': optimizer.state_dict(),
            'total_step': total_step,
            'total_train_loss': total_train_loss},
            train_args.checkpoint_path)
    dist.barrier()
    logger.info(f"Elapsed Training Time: {time.time() - t0} seconds")
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    world_size = 4
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )
