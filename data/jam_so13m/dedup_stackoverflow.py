import os
import random
import string
import pickle
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
import fire



def loopiter(partnum, files_a_part, fundats, stackoverflow_dats, test_filename, threshold, dedup_outfile, lsh_dir, partstart, partend):
    lsh_path = lsh_dir + f'/so_lsh_p{partnum}.pkl' 
    if partnum < partstart or partnum >= partend:
        return
    if os.path.isfile(lsh_path):
        lsh = pickle.load(open(lsh_path, 'rb'))
    else:
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        for filename in tqdm(files_a_part, desc=f'processing so folder part {partnum}'):
            if filename.endswith('.txt'):
                filepath = filename.split('/')[1].split('.txt')[0]
                content = stackoverflow_dats[filepath]
                minhash = MinHash(num_perm=128)
                for word in content.split():
                    minhash.update(word.encode('utf-8'))
                lsh.insert(filename, minhash)
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

def main(test_filename:str= '/nfs/projects/funcom/data/javastmt_fc/output/tdats.test',
         stackoverflow_text_id_filename:str='/sorna/datasets/jam_so13m/stackoverflow_txtfiles.pkl',
         fundats_file: str = '/sorna/datasets/jam_jm52m/fundats-j1.pkl',
         stackoverflow_text_filename:str='/sorna/datasets/jam_so13m/jam_so13m.pkl',
         lsh_dir:str='so_lsh_parts',
         dedup_outfile:str='dedup_testfids.txt',
         threshold:float=0.50,
         partstart:int =0,
         partend:int=100

        ):
    if(not os.path.exists(lsh_dir)):
        os.mkdir(lsh_dir)
    stackoverflow_file = open(stackoverflow_text_id_filename,'rb')
    files_a = pickle.load(stackoverflow_file)

    numfiles_a = len(files_a)
    nfsplit = int(numfiles_a / 100)

    files_a_split = [files_a[i:i+nfsplit] for i in range(0, numfiles_a, nfsplit)]

    fundats = pickle.load(open(fundats_file, 'rb'))
    stackoverflow_dats = pickle.load(open(stackoverflow_text_filename, 'rb'))

    for partnum, fids_a_part in enumerate(files_a_split):
        loopiter(partnum, fids_a_part, fundats, stackoverflow_dats,test_filename, threshold, dedup_outfile, lsh_dir, partstart, partend)
    #break # debug, only run once
if __name__ =='__main__':
    fire.Fire(main)
