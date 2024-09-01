import pandas as pd
from tqdm import tqdm
import json
import transformers
import csv
import os

with open('SETTINGS.json', 'r') as file:
    SETTINGS = json.load(file)

transformers.logging.set_verbosity_error()

file1 = open('unigram_freq.csv', 'r')
Lines = file1.readlines()

entries = []
for line in Lines[1:100]:
    entries.append(line.split(','))

# Instantiate the LLM
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import shutil

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config.pad_token_id = tokenizer.pad_token_id
id_eot = tokenizer.convert_tokens_to_ids(["<|eot_id|>"])[0]

manual_exclusions = ['the','of','and','to','a','in','for','is','on','that','by','this','with','i','you','it','not','or','be',
'are','from','at','as','your','all','have','new','more','an','was','we','will','can','us','about','if','my','has','but','our',
'other','do','they','he','may','what','which','their','use','any','there','only','so','his','when','here','who','also','now',
'get','pm','c','e','am','been','would','how','were','me','s','some','these','its','x','than','had','into','n','re','go','b',
'them','should','her','t','such','please','then','jan','d','where','m','r','sex','january','p','could','f','ebay','l','w','o',
'uk','g','k','y','why','shall','j']

# Create a custom Prompter object

from prompts import Prompter

class MyPrompter(Prompter):
    def is_it_valid_English(self, word):
        messages = self.sys_answerer()
        prompt = f"Is the word '{word}' a valid English word, other than an acronym or abbreviation? "
        prompt += "Reply with the single word yes or no only, with no introduction, punctuation, or other added text. "
        return self.run_prompts(messages, prompt, 0.1, False)
    def is_it_layman(self, word):
        messages = self.sys_answerer()
        prompt = f"Is the word '{word}' familiar to a layman? "
        prompt += "Reply with the single word yes or no only, with no introduction, punctuation, or other added text. "
        return self.run_prompts(messages, prompt, 0.1, False)

p = MyPrompter(model, tokenizer)

valid = []
invalid = []
for i, entry in tqdm(zip(range(0, len(entries)), entries)):
    if p.is_it_valid_English(entry[0]).strip().lower() == 'yes' and p.is_it_layman(entry[0]).strip().lower() == 'yes' \
        and entry[0] not in manual_exclusions:
        valid.append(entry)

with open(os.path.join(SETTINGS['CLEAN_DATA_DIR'], my_freq.csv'), 'w') as myfile:
    for entry in valid: 
        myfile.write(','.join(entry))

    

    
