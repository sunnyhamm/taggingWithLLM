# -*- coding: utf-8 -*-
"""
Created on Tue May 27 2025

@author: Sunny Hammett
"""

import pandas as pd
import time
import re
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
from huggingface_hub import login
import torch
# user defined models:
from api_keys import Settings  # unable to upload api keys to Git. can create the keys by go to the websites and open an account
from llm_models import *

# import MDR definitions
df_mdr = pd.read_csv('./data/cleanMDR.csv')[['name','definition']] \
    .rename(columns={'name':'mdr_name','definition':'mdr_definition'})[:100]
print(df_mdr.shape)

# import random sentences
df_ran = pd.read_csv('./data/random_sentences_1000.csv')[:100]
print(df_ran.shape)

df_mdr = df_mdr.iloc[:3] # for testing
df_ran = df_ran.iloc[:3] # for testing

################## open.ai ##########################

# limit: 3 RPM requests per minute, 200 RPD (per day)
start = time.time()

client = OpenAI(
  api_key=Settings.openai_key
)
    
df = gpt4omini(df_mdr, 'mdr_definition', client) 
df.to_csv('./data/mdr_definition_gpt4omini_parsed_delete.csv', index=False)

df = gpt4omini(df_ran, 'text', client) 
df.to_csv('./data/random_sen_gpt4omini_parsed_delete.csv', index=False)


end = time.time()
duration = end - start
minutes = int(duration // 60)
seconds = duration % 60
print(f"Execution time: {minutes} min {seconds:.2f} sec")
 

################## LLaMA3 ##########################
# reate a Function That Sends Requests to LLaMA 3
API_URL = Settings.llama3_api_url
headers = {"Authorization": Settings.llama3_api_header_auth}

start = time.time()

df = llama3(df_mdr, 'mdr_definition', API_URL, headers) 
df.to_csv('./data/mdr_definition_llama3_delete.csv', index=False)

df = llama3(df_ran, 'text', API_URL, headers) 
df.to_csv('./data/random_sen_llama3_delete.csv', index=False)


end = time.time()
duration = end - start
minutes = int(duration // 60)
seconds = duration % 60
print(f"Execution time: {minutes} min {seconds:.2f} sec")



################## flanT5large ##########################
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

start = time.time()
    
df = flan_t5_large(df_mdr, 'mdr_definition', tokenizer, model, device) 
df.to_csv('./data/mdr_definition_flan_t5_large_delete.csv', index=False)

df = flan_t5_large(df_ran, 'text', tokenizer, model, device) 
df.to_csv('./data/random_sen_flan_t5_large_delete.csv', index=False)


end = time.time()
duration = end - start
minutes = int(duration // 60)
seconds = duration % 60
print(f"Execution time: {minutes} min {seconds:.2f} sec")



################## flan-alpaca-base ##########################

# Replace with your token from https://huggingface.co/settings/tokens
login(Settings.flan_alpaca_base_login)

model_name = "declare-lab/flan-alpaca-base"  # Hugging Face model repo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

start = time.time()
    
df = flan_alpaca_base(df_mdr, 'mdr_definition', tokenizer, model, device) 
df.to_csv('./data/mdr_definition_flan_alpaca_base_delete.csv', index=False)

df = flan_alpaca_base(df_ran, 'text', tokenizer, model, device) 
df.to_csv('./data/random_sen_flan_alpaca_base_delete.csv', index=False)


end = time.time()
duration = end - start
minutes = int(duration // 60)
seconds = duration % 60
print(f"Execution time: {minutes} min {seconds:.2f} sec")



################## MISC ##########################
'''
gemini pro1.5 needs payment info, even for the free tier

mistral I ran on Census side--- I have a copy of the downloaded model;  no results produced for either MDR definitions or Random sentences

'''
