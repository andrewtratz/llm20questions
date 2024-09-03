
Below you can find a outline of how to reproduce my solution for the LLM 20 Questions competition [2024], which was entered under the Tricksy Hobbitses team name and placed second in the competition.

If you run into any trouble with the setup/code or have any questions please contact me on Kaggle at https://www.kaggle.com/jademonk.  

## ARCHIVE CONTENTS [for prize submission]

submission.tar.gz : original kaggle model upload from winning submission

README.md : this file

requirements.txt : results of pip freeze (for offline keyword preprocessing)

directory_structure.txt : full directory structure

LICENSE : open source license

entry_points.md : list of entry points (for data preprocessing)

process_keywords.py : keyword preprocessing script

prompts.py : Prompter class module

SETTINGS.json : directory used for output data

llm-20-questions-solution.ipynb: A Kaggle notebook equivalent to the final submission, with additional comments added [actual submission notebook included in submission.tar.gz]

## HARDWARE:

#### Data preprocessing instance

Ubuntu 22.04.3 LTS 

30 CPU cores, 205.4 GB RAM, 525.8 GB SSD [Lambda Labs instance]

1 x NVIDIA A100 (48 GB)

#### Submission preparation instance

Kaggle 2xT4 GPU instance

### SOFTWARE (python packages are detailed separately in `requirements.txt`):

Python 3.10.12

Nvidia Driver Version: 535.129.03   

CUDA Version: 12.2  
 
### Model permissions

Llama 3.0 requires permission from Meta to download

Go to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and request permissions for this model to be granted. Proceed once Meta approves the request.

Create a login token for Huggingface CLI at https://huggingface.co/settings/tokens


## DATA SETUP (ON PREPROCESSING INSTANCE)

### below are the shell commands used in each step, as run from the project directory

pip install -U "huggingface_hub[cli]"

huggingface-cli login
** Note: must supply valid login token at the prompt

huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct

pip install kaggle

kaggle datasets download -d rtatman/english-word-frequency

unzip *.zip

pip install -r requirements.txt  

# DATA PROCESSING

Run the following shell command from the project directory:

python process_keywords.py

This will output a file named my_freq.csv. This file can be uploaded to Kaggle as a dataset which will be included in the submission package.  

# SUBMISSION PREPARATION

1) Load llm-20-questions-solutions.ipynb into a Kaggle notebook with Internet on and 2xT4 GPUs
2) Add the following datasets to the Kaggle notebook:
   * https://www.kaggle.com/datasets/jademonk/keyword-list
   * https://www.kaggle.com/datasets/canming/llama-3-1-8b-instruct
   * https://www.kaggle.com/datasets/jademonk/frequencies
3) Run the Kaggle notebook
4) Submit the output file [named 'submission.tar.gz'] to the competition
