import tempfile
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import RobertaConfig, RobertaForMaskedLM
from torch.optim import AdamW
from tqdm import tqdm 
from transformers import get_linear_schedule_with_warmup
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# eos_token_id is not defined, since we have no tokenizer. -1 means silence, so let use 2 as eos.
# MIDI notes in this dataset are never below 21, so [0--21] are free
eos_token_id = 2
mask_token_id = 1
pad_token_id = 0
vocab_size = 128*2
max_length_sequence = 2560

def pretrain():
    # Setup for multi-GPU
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running DDP on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()

    loader_val = torch.load('dataloaders/15p_loader_val.pt')
    loader_train_1st = torch.load('dataloaders/15p_loader_train_e0.pt')

    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_length_sequence, # how far to do cross-attention (full length of sequence)
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        eos_token_id=eos_token_id,
    )

    model = RobertaForMaskedLM(config)
    model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    # Train
    epochs = 20

    ddp_model.train()
    # initialize optimizer (parameters from roberta paper)
    optim = AdamW(ddp_model.parameters(), lr=6e-4, eps=1e-6, betas=[0.9, 0.98])

    total_steps = len(loader_train_1st) * epochs
    warmup_steps = int(total_steps * 0.05)

    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Load each epochs' own train loader (different mask locations per epoch)
        loader_train = torch.load(f'dataloaders/15p_loader_train_e{epoch}.pt')

        # checkpoint model
        CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
        if rank == 0:
            print("Checkpointing", CHECKPOINT_PATH)
            torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

        # Use a barrier() to make sure that process 1 loads the model after process 0 saves it.
        dist.barrier()

        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=map_location))
    
        loop = tqdm(loader_train, leave=True)
        batch_losses = []
        for batch in loop:
            optim.zero_grad()
            
            input_ids = batch['input_ids'].to(device_id)
            attention_mask = batch['attention_mask'].to(device_id)
            labels = batch['labels'].to(device_id)

            outputs = ddp_model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            
            loss = outputs.loss
            batch_losses.append(loss.item())
            loss.backward()
            optim.step()
            scheduler.step()

            loop.set_description(f'Train Epoch {epoch+1}')
            loop.set_postfix(loss=np.mean(batch_losses))
        train_losses.append(np.mean(batch_losses))

        val_loop = tqdm(loader_val, leave=True)
        val_batch_losses = []
        for val_batch in val_loop:
            input_ids = val_batch['input_ids'].to(device_id)
            attention_mask = val_batch['attention_mask'].to(device_id)
            labels = val_batch['labels'].to(device_id)
            
            outputs = ddp_model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            
            val_loss = outputs.loss
            val_batch_losses.append(val_loss.item())

            val_loop.set_description(f'Valid Epoch {epoch+1}')
            val_loop.set_postfix(loss=np.mean(val_batch_losses))
        val_losses.append(np.mean(val_batch_losses))

    # Finish up multi-GPU
    dist.destroy_process_group()

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="valid")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss functions during pretraining")
    plt.savefig('losses.png')

    if rank == 0:
        print("Deleting tmp checkpoint and saving model")
        os.remove(CHECKPOINT_PATH)
        torch.save(ddp_model.state_dict(), 'model/model.safetensors')


if __name__=="__main__":
    pretrain()

    # Run this script with:
    '''
    torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:12485 pretrain.py
    '''