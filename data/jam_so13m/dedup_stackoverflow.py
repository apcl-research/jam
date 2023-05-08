import os
import random
import string
import pickle
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from datasets import load_dataset


# Define the paths for the A and B folders
#folder_a = '/home/cmc/dev/projects/datasets/stackoverflow/out/'
#folder_b = '../owt/funcom_test/'
file_b = '/nfs/projects/funcom/data/javastmt_fc/output/tdats.test'

stackoverflow_file = open('stackoverflow_txtfiles.pkl','rb')
files_a = pickle.load(stackoverflow_file)

numfiles_a = int(len(files_a) / 2)
nfsplit = int(numfiles_a / 100)

files_a_split = [files_a[i:i+nfsplit] for i in range(numfiles_a, len(files_a), nfsplit)]

fundats = pickle.load(open('/sorna/datasets/jam_jm52m/fundats-j1.pkl', 'rb'))
stackoverflow_dats = pickle.load(open('jam_so13m.pkl', 'rb'))
for partnum, files_a_part in enumerate(files_a_split):

    if os.path.isfile(f'so_lsh_parts/so_lsh_p{partnum}.pkl'):
        lsh = pickle.load(open(f'so_lsh_parts/so_lsh_p{partnum}.pkl', 'rb'))
    else:
        lsh = MinHashLSH(threshold=0.50, num_perm=128)
        # Generate MinHash signatures for all files in folder A
        for filename in tqdm(files_a_part, desc=f'processing so folder part {partnum}'):
            if filename.endswith('.txt'):
                filepath = filename.split('/')[1].split('.txt')[0]
                content = stackoverflow_dats[filepath]
                # Generate a MinHash signature for the file
                minhash = MinHash(num_perm=128)
                for word in content.split():
                    minhash.update(word.encode('utf-8'))
                # Add the signature to the LSH index
                lsh.insert(filename, minhash)

        with open(f'stackoverflow/so_lsh_parts/so_lsh_m_p{partnum}.pkl', 'wb') as f:
            pickle.dump(lsh, f)

# Deduplicate files in folder B
#for filename in os.listdir(folder_b):
#    if filename.endswith('.txt'):
#        filepath = os.path.join(folder_b, filename)
#        with open(filepath, 'r') as f:
#            content = f.read()
#            # Generate a MinHash signature for the file
#            minhash = MinHash(num_perm=128)
#            for word in content.split():
#                minhash.update(word.encode('utf-8'))
#            # Query the LSH index to see if there are similar files in folder A
#            matches = lsh.query(minhash)
#            if len(matches) > 0:
#                # Remove the file from folder B if there is a match in folder A
#                os.remove(filepath)
#                print(f'{filename} removed from folder B.')

    outf = open('dedup_testfids_m_th_5.txt', 'a')

    with open(file_b, 'r') as fb:
        for line in tqdm(fb):
            (fid, tdat) = line.split('<SEP>')
            fid = int(fid)

            tdat = fundats[fid]

            minhash = MinHash(num_perm=128)
            for word in tdat.split(' '):
                minhash.update(word.encode('utf-8'))
            matches = lsh.query(minhash)

            if len(matches) != 0:
                print(f'{fid}\t{matches}', file=outf)
                #print(f'removed {fid} due to {matches}')

    outf.flush()
    outf.close()

    #break # debug, only run once

