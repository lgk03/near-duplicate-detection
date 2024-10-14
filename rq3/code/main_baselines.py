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
import numpy as np

# from script/08c
def is_clone(model, distance):
    try:
        prediction = model.predict(np.array(distance).reshape(1, -1))  # 0 = near-duplicates, 1 = distinct
    except ValueError:
        prediction = [0]

    if prediction == [0]:
        return True
    else:
        return False

no_of_inferences = 0

# counter for number of inferences
def increase_no_of_infers():
    global no_of_inferences
    no_of_inferences += 1
    if no_of_inferences % 100 == 0:
        print(f"Number of inferences: {no_of_inferences}")

# counter for number of inferences
def increase_no_of_inferences():
    global no_of_inferences
    no_of_inferences += 1
    if no_of_inferences % 100 == 0:
        print(f"Number of inferences: {no_of_inferences}")

app = Flask(__name__)

# parametersJava = {'distance': x}
@app.route('/equals', methods=('GET', 'POST'))
def equal_route_rted():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json' or content_type == 'application/json; utf-8':
        data = json.loads(request.data)
    else:
        return 'Content-Type not supported!'

    parametersJava = data
    dist = parametersJava['distance']

    # classifying the distance using the trained classifier
    result = is_clone(model, dist)

    # if no_of_inferences < 10: increase_no_of_infers(); return "false"

    result = "true" if result == 1 else "false"

    print(f"Result: {result}")
    increase_no_of_inferences()
    return result

"""
run this:
python abstract_function_python/main_baselines.py --classifier across-apps-addressbook-svm-rbf-dom-rted.sav
python abstract_function_python/main_baselines.py --classifier across-apps-addressbook-svm-rbf-visual-pdiff.sav
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the Flask server for BASELINE.')
    parser.add_argument('--classifier', type=str, required=True, help='Name of the classifier file')
    args = parser.parse_args()

    # Define possible paths for the classifier
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    possible_paths = [
        os.path.join(current_dir, 'trained_classifiers', args.classifier),
        os.path.join(parent_dir, 'trained_classifiers', args.classifier),
        os.path.join(current_dir, args.classifier),
        os.path.join(parent_dir, args.classifier)
    ]

    # Try to load the classifier from the possible paths
    model = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                print(f" ++++ Loaded classifier from: {path} ++++ ")
                break
            except Exception as e:
                print(f"Error loading classifier from {path}: {e}")

    if model is None:
        print("Error: Classifier file not found. Searched in:")
        for path in possible_paths:
            print(f"- {path}")
        exit(1)

    app.run(host="0.0.0.0", port=5187, debug=False)
