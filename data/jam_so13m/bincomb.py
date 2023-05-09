import os
import glob
import numpy as np
from tqdm import tqdm
import fire

data_dir = 'bins/'


def main(data_dir = 'bins'):
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


if __name__=='__main__':
    fire.Fire(main)

