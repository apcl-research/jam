import os
import random
import string
import pickle
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from datasets import load_dataset
import fire 



def main(stackoverflow_textfilename_list:str = '/sorna/datasets/jam_so13m/stackoverflow_txtfiles.pkl',
        fundats_filename:str = '/sorna/datasets/jam_jm52m/fundats-j1.pkl',
        stackoverflow_textdata_filename:str = '/sorna/datasets/jam_so13m/jam_so13m.pkl',
        out_filename:str = 'dedup_testfids_m_th_5.txt',
        threshold:float = 0.50, # control the percentage of similarity
        test_filename:str='/nfs/projects/funcom/data/javastmt_fc/output/tdats.test',
        lsh_outdir:str='stackoverflow/so_lsh_parts'
        ):
    stackoverflow_file = open(stackoverflow_textfilename_list,'rb')
    files_a = pickle.load(stackoverflow_file)

    numfiles_a = len(files_a)
    nfsplit = int(numfiles_a / 100)

    files_a_split = [files_a[i:i+nfsplit] for i in range(0, len(files_a), nfsplit)]

    fundats = pickle.load(open(fundats_filename, 'rb'))
    funids = list(fundats.keys())
    stackoverflow_dats = pickle.load(open( stackoverflow_textdata_filename , 'rb'))
    for partnum, files_a_part in enumerate(files_a_split):
        lsh_path = lsh_outdir + f'/so_lsh_p{partnum}.pkl'
        if os.path.isfile(lsh_path):
            lsh = pickle.load(open(lsh_path, 'rb'))
        else:
            lsh = MinHashLSH(threshold=threshold, num_perm=128)
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

            with open(lsh_path, 'wb') as f:
                pickle.dump(lsh, f)


        outf = open(out_filename, 'a')
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


if __name__ == "__main__":
    main()

