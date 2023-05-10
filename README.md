# Jam: A Language Model of Java Methods

## Code for ESEC/FSE 2023 demonstration paper, A Language Model of Java Methods with Train/Test Deduplication, by Chia-Yi Su, Aakash Bansal, Vijayanta Jain, Sepideh Ghanavati, and Collin McMillan

This repository contains all the code and detailed instructions to rebuild Jam350m models and -- in our [Automatic Program Comprehension Lab](https://huggingface.co/apcl) hub. You can either go through the entire process from scratch including compiling dataset for training or just finetuning the models that we provide with the dataset that we have already compile. We also provide the scripts for deduplication on test set.

## Quick Link
- [Dataset](#dataset)
- [Model](#model)
- [To-do list]()
- [Deduplication]()
- [Train model with Java52m from scratch](#train-model-with-java52m-from-scratch)
- [Train model with Stackoverflow13m from scratch](#train-model-with-stackoverflow13m-from-scratch)
- [Fine-tuning model]()
- [Go through entire process](#go-through-entire-process)


## Dataset
We release two datasets that we use to pre-train our models. You can use the scripts that we provide to download these datasets automatically.

| Datset      | Description |Link        |
| ----------- | ----------- |------------|
| jm52m       | jm52m is a dataset we created containing 52m Java methods from 52k Java projects. The source code originated from the Merobase and Sourcerer data releases, supplemented by our own prior work in LeClair et al. It contains code uploaded to code repositories between 2008 and 2018. We then extracted every Java method from every file and project. We removed empty methods, methods from corrupt files, and methods with parsing errors       | [link]() |
| so13m       | so13m is a dataset containing 13m discussion threads from StackOverflow. The origin of the data is the StackExchange data dump from between January 2014 and December 2022. The threads cover a multitude of topics. This dataset serves as a natural language and (often) accompanying code in the domain of software engineering. Its inclusion could help downstream tasks depending on generating or understanding natural language.           | [link]() |

To download the reuiqred datasets automatically, you can run the following command. 

```
python3 download.py --repo_id={apcl/jm52m | }  --local_dir=./yourdir --repo_type=dataset
```

This will download the all the files in the repository. If you only want to download specific files, you can simply run the following command.

  ```
  python3 download.py --repo_id=apcl/java52m --download_file=True --filename=file.pkl --local_dir=./yourdir --repo_type=dataset
  ```
    --filename: the name of the file that you want to download
    --local_dir: the name of the directory that you want to put your files
    --repo_type: the type of repo that you download the file; set to dataset if you donwload files from dataset repo

Note that you only need train.bin and val.bin if you only want to pre-train your models from scratch instead of going through the entire process. You can see more details on [Train model with Java52m from scratch](#train-model-with-java52m-from-scratch) and [Train model with Stackoverflow13m from scratch](#train-model-with-stackoverflow13m-from-scratch). However, if you want to go through the entire process, you can check [Go through entire process](#go-through-entire-process) section.

## Model
We release the model that we pre-trained. 


| Model       | Description |Link        |
| ----------- | ----------- |------------|
| jam         | This model is trained on jm52m only and trained for one epoch, which is ∼300,000 iterations.| [link]()   |
| jam-so      | This model is trained on so13m only and trained for one epoch, which is ∼300,000 iterations.| [link]()   |
| jam-sojm    | This model is trained on so13m and then jm52m for one epoch each after resetting the learning rate and decay.| [link]()   |

To download the reuiqred datasets automatically, you can run the following command. 


## Train model with Java52m from scratch

## Go through entire process

## Train model with Stackoverflow13m from scratch

## To-do list
To set up your local environment, run the following command. We recommend you to use virtual environment for running the experiements.
```
pip install -r requirements.txt
```

## Dataset generation
To generate 52 millions funcom Java methods, run the following command.
  ```
  python3 data/jam_jm52m/prepare_fc_raw.py --num-proc=4 --q90testfids-file=q90testfids.pkl --fundats-file=fundats-j1.pkl
  ```
    --num-proc: number of workers in .map() call
    --q90testfids-file: funcom Java methods testset id files
    --fundats-file: Name of Java methods raw code files; It's a dictionary file with key = function id and values = raw code

The following command is for generation of 13 millions stackoverflow data
  ```
  python3 data/jam_so13m/prepare_stackoverflow.py --num-proc=4 --stackoverflow_filename=jam_so13m.pkl
  ```
    --stackoverflow_filename: Name of file for stackoverflow data; This is a dictionary file with key = post id and values = post text
After the script is done, it will have both train.bin and val.bin in either jam_jm52m or jam_so13m directory.

## Dataset Deduplication
To deduplicate the test data included in the training set, use the following command to deduplicate test data included in Java methods

```
python3 data/jam_jm52m/dedup_fctest.py --test_filename=tdats.test --lsh_dir=fc_lsh_parts --threshold=0.5 --dedup_outfile=dedup_testfids.txt --fundats_file==fundats-j1.pkl
```
    --test_filename: file name of your test file
    --lsh_dir: directory for lsh files
    --threshold: control the level similarity; 0.7 would be a good threshold for Java 52 millions methods
    --dedup_outfile: output file with function id and duplicate functions id in lists
    --fundats_file: a pickle file that is a dictionary for raw function code with key = function id and value = raw code
To deduplicate the test data included in Stackoverflow posts, use the following command.
```
python3 data/jam_so13m/dedup_stackoverflow.py --stackoverflow_textfilename_list=stackoverflow_txtfiles.pkl --fundats_filename=fundats-j1.pkl --stackoverflow_textdata_filename=jam_so13m.pkl --out_filename=dedup_testfids.txt --threshold=0.5 --test_filename=tdats.test --lsh_outdir=lsh_outdir
```
    --stackoverflow_textfilename_list: a pickle file that is a list for stackoverflow file name
    --fundats_filename: a pickle file that is a dictionary for raw function code files with key = function id and value = raw code
    --stackoverflow_textdata_filename: a pickle file that is a dictionary for stackoverflow's posts with key = post id and value = stackoverflow post
    --out_filename: output file with function id and duplicate functions id in lists
    --threshold: control the level similarity;
    --test_filename: file name of your test file
    --lsh_outdir: directory for lsh files


## Train 
If you want to train the Jam350m model from scratch and you only have one gpu, use the following command to train the model.
  ```
  python3 train.py config/{train_funcom_raw | train_stackoverflow}.py
  ```
    train_funcom_raw: traninig with 52 millions funcom Java methods

    train_stackoverflow: training with 13 millions stackoverflow posts from sratch
  
If you have multiple gpus, use the following command to train the model.
  ```
  torchrun --standalone --nproc_per_node=1 train.py config/train_funcom_raw.py --out_dir=jam350m_jm --rdzv-backend=c10d  --rdzv-endpoint=localhost:0 --nproc-per-node=1
  ```
You may want to refer to this [document](https://pytorch.org/docs/stable/elastic/run.html) to change the port number for rdzv-endpoint if you have multiple instances on the same machine. Otherwise, you will have two different training instances but updating the same model weights.

## Fine-tuning
If you want to finetune instead of training from scratch, you can run the following command. Before you train the model, you need to download the model from this [hub](https://huggingface.co/apcl/jam350m) with ``download.py``. You can download the model by simply running the following command.
```
python3 download.py --repo_id=apcl/java52m --local_dir=./yourdir --repo_type=dataset
```
After downloading the model files, you can simply run the following command to finetune the model.

```
python3 train.py config/finetune_funcom_raw.py
```
Be sure to change the ``out_dir`` in the finetune_funcom_raw.py to the same ``dir`` as the ``--local_dir``.

## Inference
To use the model that you train to infer the summary of Java methods, use the following command.
```
python sample_funcom.py --out_dir=outdir
```
  
  

  
