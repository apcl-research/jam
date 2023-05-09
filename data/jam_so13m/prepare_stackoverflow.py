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
import argparse
import bincomb

random.seed(1337)



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--stackoverflow_filename', type=str, default='/sorna/datasets/jam_so13m/jam_so13m.pkl')
    parser.add_argument('--data-dir', type=str, default='bins/')

    args = parser.parse_args()
    num_proc = args.num_proc # number of workers in .map() call # good number to use is ~order number of cpu cores // 2
    stackoverflow_filename = args.stackoverflow_filename
    data_dir = args.data_dir

    stackoverflow_file = pickle.load(open(stackoverflow_filename, 'rb'))

    contents = list(stackoverflow_file.values())

    pt = int(len(contents) * .05)
    for partnum in range(0, 1):

        print(f'starting part {partnum}')

        bin_filename = data_dir + f'val_5pt_p{partnum}.bin'
        if os.path.isfile(bin_filename):
            continue

        start_pt = (partnum * pt)
        end_pt = ((partnum+1) * pt)

        so_5pt_px = contents[start_pt:end_pt]

    
        # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
        #dataset = load_dataset('text', data_files={'train': so_5pt_px}, sample_by="document")
        dataset_dict = {"text": so_5pt_px}
        dataset = Dataset.from_dict(dataset_dict)
        shmdir = 'tmp/'
        for f in os.listdir(shmdir):
            os.remove(os.path.join(shmdir, f))

        pickle.dump(dataset, open(f'pkls/dataset_stackoverflow_5pt_p{partnum}.pkl', 'wb'))

        # owt by default only contains the 'train' split, so create a test split
        split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test') # rename the test split to val


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
            filename = os.path.join(data_dir, f'{split}_5pt_p{partnum}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

            print(f"writing {filename}...")
            idx = 0
            for example in tqdm(dset):
                arr[idx : idx + example['len']] = example['ids']
                idx += example['len']
            arr.flush()
    bincomb.main()
        # to read the bin files later, e.g. with numpy:
        # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
