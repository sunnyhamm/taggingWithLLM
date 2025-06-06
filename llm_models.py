# -*- coding: utf-8 -*-
"""
Created on Tue May 27 2025

@author: Sunny Hammett
"""

import pandas as pd
import time
import re
from tqdm import tqdm
# llm models
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Text2TextGenerationPipeline
import requests
from huggingface_hub import login

# open.ai

# LLaMA 3 (8B or 70B) 

# Gemini 1.5 Pro

# google/flan-t5-large, or T5

# declare-lab/flan-alpaca-base

# mistralai/Mistral-7B-Instruct-v0.1 (if using transformers with Accelerate/DeepSpeed)


################## open.ai ##########################

# limit: 3 RPM requests per minute, 200 RPD (per day)
# some kind of 200 requests per day limit, despite of error msg saying wait 7m and try again. doesn't work.

def gpt4omini(df, col, client): # col is the column that will be analyzed
    
    def enrich_row(text):
        prompt = f"""
        Given the following content, generate the following:
        - Name: A concise title.
        - Description: A short summary.
        - Tagging: A list of relevant tags.
        - Annotation: Detailed NLP annotation like entities and sentiment.
    
        Content:
        \"\"\"{text}\"\"\"
        """
        response = client.chat.completions.create( 
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    # Assuming df already exists and enrich_row is defined
    generated_info_list = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            enriched = enrich_row(row[col])  # OpenAI API call
            generated_info_list.append(enriched)
        except Exception as e:
            print(f"❌ Error on row {i}: {e}")
            generated_info_list.append(None)  # or "" or a default fallback
    
        time.sleep(20)  # Respect GPT-4o-mini's RPM limit (3 RPM → 1 request per 20s)
    
    # Attach results to DataFrame
    df["generated_info"] = generated_info_list
    
    def parse_generated_info(text):
        # Extract name
         name = re.search(r"\*\*Name:\*\* (.+)", text)
         name = name.group(1).strip() if name else ""
       
         # Extract description
         description = re.search(r"\*\*Description:\*\* (.+?)(?=\n- \*\*Tagging)", text, re.DOTALL)
         description = description.group(1).strip() if description else ""
       
         # Extract tagging list
         tagging = re.search(r"\*\*Tagging:\*\*([\s\S]+?)(?=\n- \*\*Annotation)", text)
         if tagging:
             tags = re.findall(r"- (.+)", tagging.group(1))
         else:
             tags = []
       
         # Extract annotation
         annotation = re.search(r"\*\*Annotation:\*\*([\s\S]+?)(?=Sentiment)", text)
         annotation_text = annotation.group(1).strip() if annotation else ""
       
         # Extract sentiment
         sentiment = re.search(r"\*\*Sentiment:\*\* (.+)", text)
         sentiment = sentiment.group(1).strip() if sentiment else ""
       
         return pd.Series({
             "name": name,
             "description": description,
             "tagging": tags,
             "annotation": annotation_text,
             "sentiment": sentiment
         })
    
    
    # Apply parsing to the DataFrame
    parsed_df = df['generated_info'].apply(parse_generated_info)
    
    # Join back to original DataFrame
    df_final = pd.concat([df, parsed_df], axis=1)
    
    return df_final


########### Gemini 1.5 pro ---  need to enter payment info (even though not going to be charged)


# import google.generativeai as genai

# genai.configure(api_key="AIzaSyBWzQgPMNovMYrfoUw2xuhA6bhhtoSv1G8")
# model = genai.GenerativeModel("gemini-1.5-pro-latest")

# def build_prompt(text):
#     return f"""
# You are a smart AI assistant. Analyze the following sentence and extract:

# - Name (if any mentioned)
# - Description (a brief interpretation)
# - Tags (keywords or themes)
# - Annotation (any context or explanation)
# - Sentiment (Positive / Negative / Neutral)

# Sentence: "{text}"

# Respond in JSON format with keys: name, description, tags, annotation, sentiment.
# """

# import time

# results = []

# for index, row in df.iterrows():
#     prompt = build_prompt(row["mdr_definition"])
    
#     try:
#         response = model.generate_content(prompt)
#         content = response.text

#         # Parse JSON from the response
#         import json
#         parsed = json.loads(content)

#         results.append(parsed)

#     except Exception as e:
#         print(f"Error on row {index}: {e}")
#         results.append({"name": None, "description": None, "tags": None, "annotation": None, "sentiment": None})

#     time.sleep(0.5)  # Optional: to avoid rate limits

# # Create a DataFrame from the results
# new_cols = pd.DataFrame(results)
# df = pd.concat([df, new_cols], axis=1)

# # df.to_csv("enriched_definitions.csv", index=False)



############## LLaMA 3;    nothing produced. -- from mdr deifnitions,  or random sentences

def llama3(df, col, API_URL, headers): # col is the column to be analyzed
    
    def call_llama3(text):
        prompt = f"""
    You are an intelligent AI that extracts structured information from product or incident definitions.
    
    Return output ONLY as JSON in this format:
    {{
      "name": "short title of the incident or concept",
      "description": "what it is about in simple terms",
      "tagging": ["tags", "related", "concepts"],
      "annotations": [{{"entity": "XYZ", "type": "MedicalTerm"}}],
      "sentiment": "Positive" / "Negative" / "Neutral"
    }}
    
    Now extract from this sentence:
    \"{text}\"
    """
    
        payload = {"inputs": prompt}
        response = requests.post(API_URL, headers=headers, json=payload)
        try:
            return response.json()[0]['generated_text']
        except Exception as e:
            return str(e)
    
    
    # apply this to df, with rate limiting
    import time
    from tqdm import tqdm
    
    results = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            output = call_llama3(row[col])
            results.append(output)
        except Exception as e:
            print(f"Error at row {i}: {e}")
            results.append(None)
        time.sleep(3)  # Adjust based on rate limits
    
    # parse the results
    import json
    
    parsed_data = []
    
    for output in results:
        try:
            data = json.loads(output)
            parsed_data.append({
                'name': data.get('name'),
                'description': data.get('description'),
                'tagging': data.get('tagging'),
                'annotations': data.get('annotations'),
                'sentiment': data.get('sentiment')
            })
        except Exception:
            parsed_data.append({
                'name': None,
                'description': None,
                'tagging': None,
                'annotations': None,
                'sentiment': None
            })
    
    parsed_df = pd.DataFrame(parsed_data)
    df = pd.concat([df.reset_index(drop=True), parsed_df], axis=1)
    
    return df



############# mistral, running off line mode. online version connection error. --- ran on Census side.
##############  some decoding error, same for mdr definition and random sentences
 

# import ollama
# import pandas as pd
# import chardet

# def build_prompt(sentence):
#     return f"""
#         Given the following sentence, extract the following:
       
#         - Name
#         - Description
#         - Tags
#         - Annotation
#         - Sentiment (Positive, Neutral, Negative)
       
#         Sentence: "{sentence}"
       
#         Respond in JSON format like:
#         {{
#           "name": "...",
#           "description": "...",
#           "tags": [...],
#           "annotation": "...",
#           "sentiment": "..."
#         }}
#         """
       
# import subprocess
# import json

# def query_ollama(prompt, model="mistral"):
#     result = subprocess.run(
#         ["ollama", "run", model, prompt],
#         capture_output=True,
#         text=True,
#         timeout=60
#     )
#     return result.stdout.strip()

# #####
# fileP = r"C:\Users\hamme040\Documents\work\ECON\sematch\data_for_testing\mdr Variables 1.csv"
# with open(fileP, 'rb') as f:
#     result = chardet.detect(f.read())
# encoding_mdr = result['encoding']

# df = pd.read_csv(fileP, encoding=encoding_mdr)


# df = pd.read_csv('./data/cleanMDR.csv')

# df = df[1000:1100]

# # Extracted info will be stored here
# extracted_data = []

# for idx, row in df.iterrows():
#     text = row['definition']
#     prompt = build_prompt(text)
   
#     try:
#         response = query_ollama(prompt)
#         data = json.loads(response)
#     except Exception as e:
#         print(f"Error at row {idx}: {e}")
#         data = {
#             "name": None,
#             "description": None,
#             "tags": None,
#             "annotation": None,
#             "sentiment": None
#         }
   
#     extracted_data.append(data)

#     if idx % 50 == 0:
#         print(f"Processed {idx} rows...")

# results_df = pd.DataFrame(extracted_data)
# final_df = pd.concat([df, results_df], axis=1)

# # Save output
# # final_df.to_csv("mistral_extracted.csv", index=False)


# tried on census side, offline mode. but model has this error.   deadend. 
# UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f in position 761: character maps to <undefined>
# Error at row 1: Expecting value: line 1 column 1 (char 0)
# Exception in thread Thread-18 (_readerthread):
# Traceback (most recent call last):
#   File "C:\Users\hamme040\Conda_Home\sematch\Lib\threading.py", line 1075, in _bootstrap_inner
#     self.run()
#   File "C:\Users\hamme040\Conda_Home\sematch\Lib\site-packages\ipykernel\ipkernel.py", line 766, in run_closure
#     _threading_Thread_run(self)
#   File "C:\Users\hamme040\Conda_Home\sematch\Lib\threading.py", line 1012, in run
#     self._target(*self._args, **self._kwargs)
#   File "C:\Users\hamme040\Conda_Home\sematch\Lib\subprocess.py", line 1601, in _readerthread
#     buffer.append(fh.read())
#                   ^^^^^^^^^
#   File "C:\Users\hamme040\Conda_Home\sematch\Lib\encodings\cp1252.py", line 23, in decode
#     return codecs.charmap_decode(input,self.errors,decoding_table)[0]
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f in position 332: character maps to <undefined>
# Error at row 2: Expecting value: line 1 column 1 (char 0)




################################## flan-t5-large no results, MDR definition or Random Sentences
def flan_t5_large(df, col, tokenizer, model, device): # col is the column to be analyzed
    
    def build_prompt(text):
        return f"""
    Extract the following from the sentence below:
    
    - Name (if mentioned)
    - Description (what it is about)
    - Tags (keywords)
    - Annotation (any insight or context)
    - Sentiment (Positive, Neutral, or Negative)
    
    Sentence: "{text}"
    """
    def run_flan_t5_inference(prompt, max_tokens=256):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    results = []
    
    for idx, row in df.iterrows():
        prompt = build_prompt(row[col])
        
        try:
            output = run_flan_t5_inference(prompt)
            results.append(output)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            results.append("")
    
        time.sleep(0.2)  # optional, not usually needed locally
    
    
    def parse_output(output):
        try:
            name = re.search(r"(?i)Name\s*:\s*(.+)", output)
            description = re.search(r"(?i)Description\s*:\s*(.+)", output)
            tags = re.search(r"(?i)Tags\s*:\s*(.+)", output)
            annotation = re.search(r"(?i)Annotation\s*:\s*(.+)", output)
            sentiment = re.search(r"(?i)Sentiment\s*:\s*(.+)", output)
    
            return {
                "name": name.group(1).strip() if name else None,
                "description": description.group(1).strip() if description else None,
                "tags": tags.group(1).strip() if tags else None,
                "annotation": annotation.group(1).strip() if annotation else None,
                "sentiment": sentiment.group(1).strip() if sentiment else None
            }
        except:
            return {"name": None, "description": None, "tags": None, "annotation": None, "sentiment": None}
    
    parsed_results = [parse_output(r) for r in results]
    new_cols = pd.DataFrame(parsed_results)
    df = pd.concat([df, new_cols], axis=1)
    
    return df




######################## flan-alpaca-base  ---  no results for either MDR definition or random sentences

def flan_alpaca_base(df, col, tokenizer, model, device): # col is the column to be analyzed

    def build_prompt(text):
        return f"""
    ### Instruction:
    Extract the following information from the sentence:
    
    - Name (if any)
    - Description (summary)
    - Tags (keywords)
    - Annotation (extra info or context)
    - Sentiment (Positive / Negative / Neutral)
    
    ### Input:
    {text}
    
    ### Response:
    """
    
    def run_flan_alpaca_inference(prompt, max_tokens=256):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    results = []
    
    for idx, row in df.iterrows():
        prompt = build_prompt(row[col])
        
        try:
            output = run_flan_alpaca_inference(prompt)
            results.append(output)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            results.append("")
    
        if idx % 100 == 0:
            print(f"Processed {idx} rows")
    
            time.sleep(0.1)  # Optional, no rate limit locally
        
        def parse_output(output):
            def extract(key):
                match = re.search(fr"{key}\s*:\s*(.*)", output, re.IGNORECASE)
                return match.group(1).strip() if match else None
        
            return {
                "name": extract("Name"),
                "description": extract("Description"),
                "tags": extract("Tags"),
                "annotation": extract("Annotation"),
                "sentiment": extract("Sentiment")
            }
        
        
        parsed_results = [parse_output(r) for r in results]
        new_cols = pd.DataFrame(parsed_results)
        df = pd.concat([df, new_cols], axis=1)
    
    return df
