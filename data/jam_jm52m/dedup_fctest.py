import os
import sys
import random
import string
import pickle
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

#import ray

#ray.init(log_to_driver=False)

#NUMTHREADS=4

# Define the paths for the A and B folders
#folder_a = '/home/cmc/dev/projects/datasets/owt/openwebtext/'
#folder_b = 'funcom_test/'
file_b = '../owt/tdats.test'

#files_a = os.listdir(folder_a)

fundats = pickle.load(open('../funcom/fundats-j1.pkl', 'rb'))
allfids = list(fundats.keys())

#fundats_r = ray.put(fundats_o)

numfids = len(allfids)
nfsplit = int(numfids / 50)

fids_a_split = [allfids[i:i+nfsplit] for i in range(0, numfids, nfsplit)]

#futures = list()

partstart = int(sys.argv[1])
partend = int(sys.argv[2])

#@ray.remote
def loopiter(partnum, fids_a_part):

    if partnum < partstart or partnum >= partend:
        return

#    fundats = ray.get(fundats_r)

    if os.path.isfile(f'fc_lsh_parts/fc_lsh_p{partnum}.pkl'):
        lsh = pickle.load(open(f'fc_lsh_parts/fc_lsh_p{partnum}.pkl', 'rb'))
    else:
        lsh = MinHashLSH(threshold=0.70, num_perm=128)
        # Generate MinHash signatures for all files in folder A
        for fid in tqdm(fids_a_part, desc=f'processing fids part {partnum}'):
            content = fundats[fid]
            # Generate a MinHash signature for the file
            minhash = MinHash(num_perm=128)
            for word in content.split():
                minhash.update(word.encode('utf-8'))
            # Add the signature to the LSH index
            lsh.insert(fid, minhash)

        with open(f'/home/cmc/dev/projects/datasets/funcom/fc_lsh_parts/fc_lsh_p{partnum}.pkl', 'wb') as f:
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

    outf = open('dedup_testfids.txt', 'a')

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

for partnum, fids_a_part in enumerate(fids_a_split):
    loopiter(partnum, fids_a_part)

#    futures.append(loopiter.remote(partnum, fids_a_part))

#    if len(futures) >= NUMTHREADS:
#        for future in futures:
#            ray.get(future)
#        futures = list()


