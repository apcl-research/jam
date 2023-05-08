# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from datasets import Dataset

import pickle
import random

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 4

fundats = pickle.load(open('fundats-j1.pkl', 'rb'))

fundats_fids = list(fundats.keys())

#random.shuffle(arr)
pt = int(len(fundats_fids) * .02)

#fundats_ds = dict()
#fundats_ds['train'] = list()
#fundats_ds['train'] = dict()
#fundats_ds['train']['id'] = list()
#fundats_ds['train']['tokens'] = list()

q90testfids = pickle.load(open('q90testfids.pkl', 'rb'))

for partnum in range(0, 50):

    print(f'starting part {partnum}')

    txtfiles = list()

    if os.path.isfile(f'bins/val_2pt_p{partnum}.bin'):
        continue

    start_pt = (partnum * pt)
    end_pt = ((partnum+1) * pt)

    fundats_fids_2pt_px = fundats_fids[start_pt:end_pt]

    for fid in tqdm(fundats_fids_2pt_px):
        #fundats_ds['train']['id'].append(fid)
        #fundats_ds['train']['tokens'].append(fundats[fid])
        #fundats_ds['train'].append(fundats[fid])

        if fid in q90testfids:
            continue

        with open(f'tmp/{fid}', 'w') as f:
            f.write(fundats[fid])

        txtfiles.append(f'tmp/{fid}')


    #dataset = Dataset.from_dict(fundats_ds)

    #txtfiles = list()
    #for file in arr:
    #    if file.endswith('.txt'):
    #        txtfiles.append(txtdir + file)

    #pickle.dump(txtfiles, open('txtfiles_30pt.pkl', 'wb'))

    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    #dataset = load_dataset("openwebtext")
    dataset = load_dataset('text', data_files={'train': txtfiles}, sample_by="document")

    shmdir = 'tmp/'
    for f in os.listdir(shmdir):
        os.remove(os.path.join(shmdir, f))

    pickle.dump(dataset, open(f'pkls/dataset_funcom_2pt_p{partnum}.pkl', 'wb'))

    #dataset = pickle.load(open('dataset_stackoverflow.pkl', 'rb'))

    #dataset = dataset.to_pandas()

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    #split_dataset = pickle.load(open('split_dataset_stackoverflow.pkl', 'rb'))
    #split_dataset['val'] = split_dataset.pop('test')

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'])
        filename = os.path.join('bins/', f'{split}_2pt_p{partnum}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        print(f"writing {filename}...")
        idx = 0
        for example in tqdm(dset):
            arr[idx : idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
