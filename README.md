# Jam: A Language Model of Java Methods

## Code for ESEC/FSE 2023 demonstration paper, A Language Model trained on Java Methods with Train/Test Deduplication 

Presented by:
- [Chia-Yi Su](https://chiayisu.github.io)
- [Aakash Bansal](https://sites.nd.edu/abansal)
- [Vijayanta Jain](https://sites.google.com/maine.edu/vijayantajain/home)
- [Sepideh Ghanavati](https://www.sepidehghanavati.com)
- [Collin McMillan](https://www3.nd.edu/~cmc/)

This repository contains all the code and detailed instructions to rebuild Jam models in our HuggingFace [Automatic Program Comprehension Lab](https://huggingface.co/apcl) hub. You can either go through the entire process from scratch including tokenization of raw source code data or just finetuning the models that we provide with the dataset that we provide as tokenized bins. We also provide the scripts for deduplication of any future test sets.

Need help?  Contact us via our [Discord Server](https://discord.gg/Sj8TwGJv4w)

## Acknowledgement
We thank Andrej Karpathy and Daniel Grittner for their work providing the NanoGPT and NanoGPT-LoRA code. This repository forks from Daniel Grittner's [NanoGPT-LoRA](https://github.com/danielgrittner/nanoGPT-LoRA) repository, which is a forked from the original [NanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. 

## Quick Link
- [To-do list](#to-do-list)
- [Pre-trained Model Checkpoints](#model)
- [Fine-tuning](#fine-tuning)
- [Deduplication toolkit](#dataset-deduplication)
- [Inference](#inference)
- [Dataset](#dataset)
- [Entire process](#entire-process)
- [Re-Training](#re-training)

## To-do list
To set up your local environment, run the following command. We recommend the use of a virtual environment for running the experiements.
```
pip install -r requirements.txt
``` 
- **If you only want to finetune one of our pre-trained models**, please see [Pre-trained Model Checkpoints](#model), [Fine-tuning](#fine-tuning), and [Inference](#inference). Additionally recommend using [Deduplication toolkit](#dataset-deduplication) before inference on your own test set.
- If you only want to deduplicate your dataset, refer to subsection 6.3, please see [Deduplication toolkit](#dataset-deduplication).
- If you want to re-train a model using our processed and tokenized dataset, please see [Retraining](#re-training)
- if you want to scratch-train, by reprocessing the dataset, pleasde see [Entire process](#entire-process) and [Re-Training](#re-Training)


## Model
We release the model that we pre-trained. 

| Model       | Description |Link        |
| ----------- | ----------- |------------|
| jam         | This model is trained on jm52m only and trained for one epoch, which is ∼300,000 iterations.| [link](https://huggingface.co/apcl/jam)   |
| jam-so      | This model is trained on so13m only and trained for one epoch, which is ∼300,000 iterations.| [link](https://huggingface.co/apcl/jam_so)   |
| jam-sojm    | This model is trained on so13m and then jm52m for one epoch each after resetting the learning rate and decay.| [link](https://huggingface.co/apcl/jam_sojm)   |

Likewise, you can use the script that we provide to download the model that we trained for fine-tuning or applications.
```
python3 download.py --repo_id=apcl/jam --repo_type=model
```
Please replace the --repo_id tag with the ``apcl/jam_so`` or ``apcl/jam_sojm`` depending on the model you wish to finetune.

## Fine-tuning

These steps will show you how to fine-tune for the code summarization application from our paper.  You can hack these scripts to do whatever you need for your own task.

### Step 1: Download the finetuning dataset

| Datset      | Description |Link        |
| ----------- | ----------- |------------|
| funcom-java-long  | is a dataset for source code summarization by Bansal et al. made available pre-publication we use to fine tune and test our model for source code summarization. This dataset is also annotate as "q90" in our scripts | [link](https://huggingface.co/datasets/apcl/funcom-java-long/tree/main)


Please cite the use of the funcom-java-long dataset as follows:
```
@article{bansal2023human,
	title={Towards modeling human attention from eye movements for neutral source code summarization},
	author={Bansal, Aakash and Sharif, Bonita and McMillan, Collin},
	journal={Proceedings of ACM Human-Computer Interaction, Vol. 7},
	year={2023}

```

### Step 2: Fine-tune model
```
python3 train.py config/finetune_funcom.py
```
Note that you need to change the ``out_dir`` in the finetune_funcom.py to the same ``dir`` as your ``--local_dir``.

## Dataset Deduplication

### Step 1: Download test set and extract it
We release our test set as a ``.tar.gz`` file in [apcl/funcom-java-long](https://huggingface.co/datasets/apcl/funcom-java-long/tree/main) repository. You can simiply run the following command to download and extract test set for inference.
```
python3 download_extract_file.py 
```
    --repo_id: the id of repository that you want to download files
    --local_dir: directory that you want to put your files
    --filename: name of the file that you want to download
We have already set the default parameters to the required parameters for downloading test set. If you just want to download and extract test set, you only need to run the command above.

### Step 2: Deduplication
To deduplicate the test data included in the training set, use the following command to deduplicate test data included in Java methods

```
python3 data/jam_jm52m/dedup_fctest.py --test_filename=tdats.test --lsh_dir=fc_lsh_parts --threshold=0.5 --dedup_outfile=dedup_testfids.txt --fundats_file==fundats-j1.pkl
```
    --test_filename: file name of your test file
    --lsh_dir: directory for lsh files
    --threshold: control the level similarity; 0.7 would be a good threshold for Java 52 millions methods
    --dedup_outfile: output file with function id and duplicate functions id in lists
    --fundats_file: a pickle file that is a dictionary for raw function code with key = function id and value = raw code
    --partstart: separate deduplication into several programs to speed up; minimum value 0
    --partend: separate deduplication into several programs to speed up; maximum value 50
To deduplicate the test data included in Stackoverflow posts, use the following command.
```
python3 data/jam_so13m/dedup_stackoverflow.py --stackoverflow_text_id_filename=stackoverflow_txtfiles.pkl --fundats_file=fundats-j1.pkl --stackoverflow_text_filename=jam_so13m.pkl --dedup_outfile=dedup_testfids.txt --threshold=0.5 --test_filename=tdats.test
```
    --stackoverflow_text_id_filename: a pickle file that is a list for stackoverflow file name
    --fundats_file: a pickle file that is a dictionary for raw function code files with key = function id and value = raw code
    --stackoverflow_text_filename: a pickle file that is a dictionary for stackoverflow's posts with key = post id and value = stackoverflow post
    --dedup_outfile: output file with function id and duplicate functions id in lists
    --threshold: control the level similarity;
    --test_filename: file name of your test file
    --lsh_outdir: directory for lsh files
    --partstart: separate deduplication into several programs to speed up; minimum for partstart = 0
    --partend: separate deduplication into several programs to speed up; maximum for partstart = 100
    
## Inference

After you download and deduplicate the test set, you can simiply run command below for inference.

```
python sample_funcom.py --out_dir=outdir
```
    --outdir: directory of the model that you want to use for inference

## Dataset
We release two datasets that we use to pre-train our models. You can use the scripts that we provide to download these datasets automatically.

| Datset      | Description |Link        |
| ----------- | ----------- |------------|
| jm52m       | jm52m is a dataset we created containing 52m Java methods from 52k Java projects. The source code originated from the Merobase and Sourcerer data releases, supplemented by our own prior work in [LeClair et al.](https://arxiv.org/abs/1904.02660) It contains code uploaded to code repositories between 2008 and 2018. We then extracted every Java method from every file and project. We removed empty methods, methods from corrupt files, and methods with parsing errors       | [link](https://huggingface.co/datasets/apcl/jm52m) |
| so13m       | so13m is a dataset containing 13m discussion threads from StackOverflow. The origin of the data is the StackExchange data dump from between January 2014 and December 2022. The threads cover a multitude of topics. This dataset serves as a natural language and (often) accompanying code in the domain of software engineering. Its inclusion could help downstream tasks depending on generating or understanding natural language.           | [link](https://huggingface.co/datasets/apcl/so13m) |

To download the required datasets automatically, you can run the following command. 

```
python3 download.py --repo_id=apcl/jm52m  --local_dir=./data/yourdir --repo_type=dataset
```

This will download the all the files in the repository. If you only want to download specific files, you can simply run the following command.  **Note:** The above command will download the entire dataset including the deduplication files (total 200gb+).  If you just want the raw data, use the --filename parameter like in the next command.  Specific files you might want for the raw data are ``fundats-j1.json.gz`` and ``jm52m.sql.gz``.  Or you may wish to retrain your own Jam models using our processed and tokenized data in ``train.bin`` and ``val.bin``.

Our raw data includes full traceability from the methods to the files and projects where those methods originate.  The code for each method is in ``fundats-j1`` as a Python dictionary.  You may download either the json or the pickle version.  The key for each method is an ID number.  That ID number is the ``id`` field of the ``functionalunits`` table in the SQL dump.  Other fields should be self-explanatory.

  ```
  python3 download.py --repo_id=apcl/jm52m --filename=file.pkl --local_dir=./data/yourdir --repo_type=dataset
  ```
    --repo_id: either apcl/jm52m or apcl/so13m; apcl/jm52m is for 52 million Java methods and apcl/so13m is for 13 million stackoverflow posts.
    --filename: the name of the file that you want to download
    --local_dir: the name of the directory that you want to put your files
    --repo_type: the type of repo that you download the file; set to dataset if you donwload files from dataset repo

Again, you only need ``train.bin`` and ``val.bin`` if you only want to build your Jam models from scratch instead of going through the entire process. You can see more details on [Re-Training](#re-training). However, if you want to go through the entire process, you can check [Entire process](#entire-process) section.

## Entire process
To go through the entire process, you will need an extra step to generate the ``bin`` files by your own and use these files to train your own models. 

### Step1: Dataset generation
To generate 52 millions funcom Java methods, run the following command.
  ```
  python3 data/jam_jm52m/prepare_fc_raw.py --num-proc=4 --q90testfids-file=q90testfids.pkl --fundats-file=fundats-j1.pkl
  ```
    --num-proc: number of workers in .map() call
    --q90testfids-file: funcom Java methods test set ID files
    --fundats-file: Name of Java methods raw code files; It's a dictionary file with key = function id and values = raw code
  
  You will need to download q90testfids.pkl for Java methods' ID on test set and fundats-j1.pkl as Java methods' raw code. You can download these two files in [apcl/jm52m](https://huggingface.co/datasets/apcl/jm52m) repository. You may want to refer to [Dataset](#dataset) section to see how to download these files with the script that we release.

You can run the following command to generate 13 millions Stackoverflow posts data.
  ```
  python3 data/jam_so13m/prepare_stackoverflow.py --num-proc=4 --stackoverflow_filename=jam_so13m.pkl
  ```
    --stackoverflow_filename: Name of file for stackoverflow data; This is a dictionary file with key = post id and values = post text
After the script is done, it will have both ``train.bin`` and ``val.bin`` in either ``data/jam_jm52m`` or ``data/jam_so13m`` directory. Be sure to move it to the same directory as ``train.py``.

### Step2: Train models
After generation of ``bin`` files, you can refer to step 2 of [Re-Training](#re-training) section for training your models.

## Re Training
### Step 1: Download bin files
 You will need both ``train.bin`` and ``val.bin`` to train your models. ``bin`` files can be downloaded in the following command.
  ```
  python3 download.py --repo_id=apcl/jm52m --filename={train.bin | val.bin} --local_dir=./data/yourdir --repo_type=dataset
  ```
  Note that you will need to put these two files into the same directory as ``train.py``.
### Step 2: Train models
If you want to train your own models from scratch and you only have one gpu, use the following command to train the model. 
  ```
  python3 train.py config/{train_funcom_raw | train_stackoverflow}.py
  ```
    train_funcom_raw: traninig with 52 millions funcom Java methods

    train_stackoverflow: training with 13 millions stackoverflow posts from sratch
  
If you have multiple gpus, use the following command to train the model.
  ```
  torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=1 train.py config/train_funcom_raw.py --out_dir=jam350m_jm
  ```
You may want to refer to this [document](https://pytorch.org/docs/stable/elastic/run.html) to change the port number for rdzv-endpoint if you have multiple instances on the same machine. Otherwise, you will have two different training instances but updating the same model weights.

## Citations

If you use this work in an academic paper, please cite the following:

```
@inproceedings{su2023language,
      title={A Language Model of Java Methods with Train/Test Deduplication}, 
      author={Chia-Yi Su and Aakash Bansal and Vijayanta Jain and Sepideh Ghanavati and Collin Mcmillan},
      month={December},
      year={2023},
      booktitle={Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
      location = {San Francisco, CA, USA},
      series = {ESEC/FSE 2023}
}
```

PDF available here: https://arxiv.org/abs/2305.08286

## Hardware Disclaimer
We recommend a GPU of the [NVidia Ampere architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/) or newer, because the "bfloat16" format is essential for efficient computation with our scripts. For GPUs older than that, "float32" format can be used. However, the VRAM requirements are higher in that format and computations are slower.  We used Nvidia A5000 GPUs, though an A4000 should be sufficient.

