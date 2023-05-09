import os
import sys
import random
import string
import pickle
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
import fire

def loopiter(partnum, fids_a_part, test_filename, lshdir, threshold, dedup_outfile, fundats):

    #if partnum < partstart or partnum >= partend:
    #    return

    lsh_path = lshdir + f'/fc_lsh_p{partnum}.pkl' 
    if os.path.isfile(lsh_path):
        lsh = pickle.load(open(lsh_path, 'rb'))
    else:
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        # Generate MinHash signatures for all files in folder A
        for fid in tqdm(fids_a_part, desc=f'processing fids part {partnum}'):
            content = fundats[fid]
            # Generate a MinHash signature for the file
            minhash = MinHash(num_perm=128)
            for word in content.split():
                minhash.update(word.encode('utf-8'))
            # Add the signature to the LSH index
            lsh.insert(fid, minhash)

        with open(lsh_path, 'wb') as f:
            pickle.dump(lsh, f)

    outf = open(dedup_outfile, 'a')

    with open(test_filename, 'r') as fb:
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

    outf.flush()
    outf.close()



def main(test_filename:str='/nfs/projects/funcom/data/javastmt_fc/output/tdats.test', # name of your test files for deduplication
        lsh_dir:str = 'fc_lsh_parts',
        threshold:float=0.50,  
        dedup_outfile:str='dedup_testfids.txt',
        fundats_file: str = '/sorna/datasets/jam_jm52m/fundats-j1.pkl'
        ):
    

    fundats = pickle.load(open(fundats_file, 'rb'))
    allfids = list(fundats.keys())


    numfids = len(allfids)
    nfsplit = int(numfids / 50)

    fids_a_split = [allfids[i:i+nfsplit] for i in range(0, numfids, nfsplit)]

    for partnum, fids_a_part in enumerate(fids_a_split):
        loopiter(partnum, fids_a_part, test_filename, lsh_dir, threshold, dedup_outfile, fundats)


if __name__=='__main__':
    fire.Fire(main)

