import json
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from utils import Dataset

def preprocess(data_split, max_length_sequence, mask_percentage, pad_token_id, mask_token_id):
        # We can visualize each timestep with four notes as a 'sentence', so adding an <eos> token at the end.
        n_sequences = len(data_split)
        n_notes = 4 # one exception, but we handle that later

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
        random_mask = np.random.rand(*labels.shape) < mask_percentage
        input_ids[random_mask] = mask_token_id

        print("first 100 input_ids (with noise) of the first sequence:")
        print(input_ids[0][0:100])

        encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
        return encodings


if __name__ == "__main__":
    data = json.load(open('data/jsb-chorales-16th.json'))
    pad_token_id = 0
    mask_token_id = 1

    num_epochs = 20
    mask_percentage = 0.15

    # n_notes (4) * longest sequence in the dataset
    max_length_sequence = 4*max(max(len(seq) for seq in data_split) for data_split in [data['train'], data['valid'], data['test']])
    print(max_length_sequence)

    # Save one validation dataloader
    encodings_val = preprocess(data['valid'], max_length_sequence, mask_percentage, pad_token_id, mask_token_id)
    dataset_val = Dataset(encodings_val)
    loader_val = DataLoader(dataset_val, batch_size=1, shuffle=True)
    torch.save(loader_val, f'dataloaders/{int(mask_percentage*100)}p_loader_val.pt')

    print("Mask percentage = ", mask_percentage)
    print(f"Generating {num_epochs} training dataloaders")

    # Save several training dataloaders, each epoch needs to have different masks
    for i in range(num_epochs):
        encodings_train = preprocess(data['train'], max_length_sequence, mask_percentage, pad_token_id, mask_token_id)

        dataset_train = Dataset(encodings_train)
        loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)

        torch.save(loader_train, f'dataloaders/{int(mask_percentage*100)}p_loader_train_e{i}.pt')