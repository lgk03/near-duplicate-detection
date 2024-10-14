import json
import re
import pickle
import os
import utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import pandas as pd
from flask import Flask, request
import argparse

no_of_inferences = 0

hf_model_name = None

# counter for number of inferences
def increase_no_of_inferences():
    global no_of_inferences
    no_of_inferences += 1
    if no_of_inferences % 100 == 0:
        print(f"Number of inferences: {no_of_inferences}")

def load_model_and_tokenizer(hf_model_name=''):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)
    model.eval()  # set model into evaluation mode

    feature = ''
    if 'content_tags' in hf_model_name:
        feature = 'content_tags'
    elif 'tags' in hf_model_name:
        feature = 'tags'
    elif 'content' in hf_model_name:
        feature = 'content'

    return model, tokenizer, feature

app = Flask(__name__)

seen_states = {}

# call to route /equals executes equalRoute function
# use URL, DOM String, Dom content and DOM syntax tree as params
@app.route('/equals', methods=('GET', 'POST'))
def equal_route():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json' or content_type == 'application/json; utf-8':
        fixed_json = utils.fix_json(request.data.decode('utf-8'))
        if fixed_json == "Error decoding JSON":
            print("Exiting due to JSON error")
            exit(1)
        data = json.loads(fixed_json)
    else:
        return 'Content-Type not supported!'

    # get params sent by java
    parametersJava = data

    obj1 = parametersJava['dom1']
    obj2 = parametersJava['dom2']

    if obj1 in seen_states:seen_states[obj1] += 1
    else: seen_states[obj1] = 1
    if obj2 in seen_states: seen_states[obj2] += 1
    else: seen_states[obj2] = 1
    if seen_states[obj1] % 100 == 0: print("Saw this state very often")
    if seen_states[obj2] % 100 == 0: print("Saw this state very often")

    if obj1 == obj2: return "true" # direct clones, saves compute but remove for final evaluation
    # if no_of_inferences < 15: increase_no_of_inferences();return "false" # remove for final evaluation

    # compute equality of DOM objects
    result = utils.bert_equals(obj1, obj2, model, tokenizer, feature)

    result = "true" if result == 1 else "false"

    print(f"Result: {result}")

    increase_no_of_inferences()
    return result

"""
run this:
python abstract_function_python/main.py --model "lgk03/distilBERT-NDD.html.content"
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the Flask server for the BERT model.')
    parser.add_argument('--model', type=str, default=None, help='Path to the Hugging Face model')
    args = parser.parse_args()
    hf_model_name = args.model
    print(f"******* We are using the model: {hf_model_name} *******")
    if not hf_model_name.startswith('lgk03/'):
        raise ValueError("The model name should start with 'lgk03/' | example: 'lgk03/ACROSSAPPS_NDD-ppma_test-content_tags'")

    model, tokenizer, feature = load_model_and_tokenizer(hf_model_name=hf_model_name)

    app.run(host="0.0.0.0", port=5187, debug=False)
