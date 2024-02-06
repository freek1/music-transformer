import json
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from transformers import RobertaConfig, RobertaForMaskedLM
from torch.optim import AdamW
from tqdm import tqdm 
from transformers import get_linear_schedule_with_warmup
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# eos_token_id is not defined, since we have no tokenizer. -1 means silence, so let use 2 as eos.
# MIDI notes in this dataset are never below 21, so those are free
eos_token_id = 2
mask_token_id = 1
pad_token_id = 0
vocab_size = 128*2

def get_data(rank):
    data = json.load(open('data/jsb-chorales-16th.json'))

    # n_notes (4) * longest sequence in the dataset
    max_length_sequence = 4*max(max(len(seq) for seq in data_split) for data_split in [data['train'], data['valid'], data['test']])
    if rank == 0:
        print(max_length_sequence)

    encodings_train = preprocess(data['train'], rank, max_length_sequence, pad_token_id, mask_token_id)
    encodings_val = preprocess(data['valid'], rank, max_length_sequence, pad_token_id, mask_token_id)

    dataset_train = Dataset(encodings_train)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=8, shuffle=True)

    dataset_val = Dataset(encodings_val)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True)

    return loader_train, loader_val, vocab_size, max_length_sequence
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}

def preprocess(data_split, rank, max_length_sequence, pad_token_id, mask_token_id):
        # We can visualize each timestep with four notes as a 'sentence', so adding an <eos> token at the end.
        n_sequences = len(data_split)
        n_notes = 4 # one exception, but we handle that later

        # input_ids should be [n_sequences, n_timesteps * (n_notes + 1)]
        # we append the eos tokens ad hoc
        labels = np.ones([n_sequences, max_length_sequence], dtype=int) * pad_token_id
        if rank == 0:
            print(labels.shape)

        start_time = time.time()
        for i_seq in range(n_sequences):
            n_timesteps = len(data_split[i_seq])
            for i_time in range(n_timesteps):
                n_notes = len(data_split[i_seq][i_time])
                if n_notes != 4:
                    break
                for i_note in range(n_notes):
                    # print(i_note)
                    # if i_note == 3:
                    #     # end of sentence reached, i.e.: after four notes in a timestep.
                    #     labels = np.append(labels, eos_token_id)
                    # else:
                    labels[i_seq, i_time * n_notes + i_note] = int(data_split[i_seq][i_time][i_note])
        end_time = time.time()
        if rank == 0:
            print(f"Time to process: {end_time - start_time:.2f} seconds")

        mask = labels != 0
        input_ids = np.copy(labels)

        # Now, mask 15% of the labels randomly to use as input
        random_mask = np.random.rand(*labels.shape) < 0.15
        input_ids[random_mask] = mask_token_id

        if rank == 0:
            print("first 100 input_ids (with noise) of the first sequence:")
            print(input_ids[0][0:100])

        encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
        return encodings

def pretrain():
    # Setup for multi-GPU
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()

    loader_train, loader_val, vocab_size, max_length_sequence = get_data(rank)

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

    total_steps = len(loader_train) * epochs
    warmup_steps = int(total_steps * 0.05)  # 10% of total steps for warmup

    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
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
    torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:12485 main.py
    '''