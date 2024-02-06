import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from transformers import RobertaConfig, RobertaForMaskedLM
from torch.optim import AdamW
from tqdm import tqdm 

data = json.load(open('data/jsb-chorales-16th.json'))
vocab_size = 128 + 3 # 128 notes + 3 extra tokens
max_length_sequence = 2304 # largest sequence of val set

def preprocess(data_split):
    # We can visualize each timestep with four notes as a 'sentence', so adding an <eos> token at the end.
    n_sequences = len(data_split)
    n_notes = 4 # one exception
    max_timesteps = max(len(sequence) for sequence in data_split)

    # eos_token_id is not defined, since we have no tokenizer. -1 means silence, so let use -2 as eos.
    eos_token_id = -2
    mask_token_id = 1
    pad_token_id = 0

    vocab_size = 128 + 3 # 128 notes + 3 extra tokens

    # input_ids should be [n_sequences, n_timesteps * (n_notes + 1)]
    # we append the eos tokens ad hoc
    labels = np.ones([n_sequences, max_length_sequence], dtype=int) * pad_token_id
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
    print(f"Time to process: {end_time - start_time:.2f} seconds")

    mask = labels != 0
    input_ids = np.copy(labels)

    # Now, mask 15% of the labels randomly to use as input
    random_mask = np.random.rand(*labels.shape) < 0.15
    input_ids[random_mask] = mask_token_id

    print("first 100 input_ids (with noise) of the first sequence:")
    print(input_ids[0][0:100])

    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    return encodings

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

encodings_train = preprocess(data['train'])
encodings_val = preprocess(data['valid'])

dataset_train = Dataset(encodings_train)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True)

dataset_val = Dataset(encodings_val)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=True)

config = RobertaConfig(
    vocab_size=vocab_size,
    max_position_embeddings=max_length_sequence, # how far to do cross-attention (full length of sequence)
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

model = RobertaForMaskedLM(config)

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
device= torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}.")
model.to(device)

# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-4)

epochs = 2

train_losses = []
val_losses = []

for epoch in range(epochs):
    loop = tqdm(loader_train, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        train_losses.append(loss.item())
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

    val_loop = tqdm(loader_val, leave=True)
    for val_batch in val_loop:
        input_ids = val_batch['input_ids'].to(device)
        attention_mask = val_batch['attention_mask'].to(device)
        labels = val_batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        val_loss = outputs.loss
        val_losses.append(loss.item())

        loop.set_description(f'Val epoch {epoch}')
        loop.set_postfix(loss=loss.item())

plt.plot(train_losses)
plt.plot(val_losses)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.savefig('losses.png')

# save model
model.save_pretrained('model')
