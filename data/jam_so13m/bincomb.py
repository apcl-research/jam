import os
import glob
import numpy as np
from tqdm import tqdm

import torch

#data_dir = os.path.join('data', dataset)
data_dir = 'bins/'
#train_data_p0 = np.memmap(os.path.join(data_dir, 'train_5pt_p0.bin'), dtype=np.uint16, mode='r')
#train_data_p1 = np.memmap(os.path.join(data_dir, 'train_5pt_p1.bin'), dtype=np.uint16, mode='r')

#print(train_data_p0.shape)
#print(train_data_p1.shape)

#comb = np.concatenate((train_data_p0, train_data_p1))

bins = list()

for f in tqdm(glob.glob('bins/val*')):
    val = np.memmap(f, dtype=np.uint16, mode='r')
    bins.append(val)

comb = np.concatenate(bins)

out = np.memmap('val.bin', dtype=np.uint16, mode='w+', shape=comb.shape)
out[:] = comb[:]

for f in tqdm(glob.glob('bins/train*')):
    train = np.memmap(f, dtype=np.uint16, mode='r')
    bins.append(train)

comb = np.concatenate(bins)

out = np.memmap('train.bin', dtype=np.uint16, mode='w+', shape=comb.shape)
out[:] = comb[:]

#val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


#for split, dset in tokenized.items():
#    arr_len = np.sum(dset['len'])
#    filename = os.path.join('', f'train.bin')
#    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
#    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

#    print(f"writing {filename}...")
#    idx = 0
#    for example in tqdm(dset):
#        arr[idx : idx + example['len']] = example['ids']
#        idx += example['len']
#    arr.flush()

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


