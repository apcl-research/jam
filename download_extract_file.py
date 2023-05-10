import tarfile
from huggingface_hub import hf_hub_download, snapshot_download
import fire


def main(repo_id:str='apcl/funcom-java-long', 
        local_dir:str ='./',
        filename:str='funcom_test.tar.gz'):
    hf_hub_download(repo_id=repo_id, local_dir=local_dir, filename = filename, repo_type='dataset' )
    fname = local_dir + filename
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()

if __name__ == '__main__':
    fire.Fire(main)
