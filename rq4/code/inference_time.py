from inference_utils import *
import pandas as pd
import time
from gensim.models import Doc2Vec
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# setthe technique used 
baseline = 'webembed' # bert-base, bert-adj, rted, webembed
representation = 'content'

apps = ['addressbook', 'claroline', 'ppma', 'mrbs', 'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic']
app_time = {app : 0 for app in apps} # store the avg time taken for each app

SS = pd.read_csv(f'{dataset_base}/SS.csv')
SS_sampled = SS.sample(n=10000, random_state=42) # sample 10k rows from SS

# save the sampled rows, for rted baseline (see RTEDUtils.java)
# SS_sampled.to_csv('/Users/lgk/Downloads/selected_rows.csv')
# exit()

if baseline == 'bert-base' or baseline == 'bert-adj':
    for app in apps:
        total_time = 0
        model_path = f'lgk03/WITHINAPPS_NDD-{app}_test-{representation}{-'CWAdj' if baseline == 'bert-adj' else ''}'
        model, tokenizer = load_model_and_tokenizer(model_path) 
        app_dataset = SS_sampled[SS_sampled['appname'] == app]
        for index, row in app_dataset.iterrows():
            data1 = load_state(appname=app, state=row['state1'], representation=representation, baseline=baseline)
            data2 = load_state(appname=app, state=row['state2'], representation=representation)
            start_time = time.perf_counter()
            processed_inputs = preprocess_for_inference(data1, data2, tokenizer)
            predicted = get_prediction(model, processed_inputs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
        app_time[app] = total_time / len(app_dataset)
        print(f'{app} done: {app_time[app]:.2f} seconds')

elif baseline == 'rted':
    print('RTED - see java files')


elif baseline == 'webembed':
    doc2vec_model_content = Doc2Vec.load('/Users/lgk/git/uni/web_test_generation/neural-embeddings-web-testing/trained_model/DS_content_modelsize100epoch31.doc2vec.model') # path to doc2vec model
    for app in apps:
        classifier_path = f'/Users/lgk/git/uni/web_test_generation/neural-embeddings-web-testing/trained_classifiers/within-apps-{app}-svm-rbf-doc2vec-distance-content.sav'
        model = pickle.load(open(classifier_path, 'rb'))
        total_time = 0
        app_dataset = SS_sampled[SS_sampled['appname'] == app]
        for index, row in app_dataset.iterrows():
            data1 = load_state(appname=app, state=row['state1'], representation=representation, baseline=baseline)
            data2 = load_state(appname=app, state=row['state2'], representation=representation, baseline=baseline)
            start_time = time.perf_counter()
            emb_dom1 = doc2vec_model_content.infer_vector(data1).reshape(1, -1)
            emb_dom2 = doc2vec_model_content.infer_vector(data2).reshape(1, -1)
            sim = cosine_similarity(emb_dom1, emb_dom2)
            dist = np.array([sim[0, 0]])   
            dist = dist.reshape(1, -1)
            word2vec = model.predict(dist)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
        app_time[app] = total_time / len(app_dataset)
        print(f'{app} done: {app_time[app]:.2f} seconds')

print_report(app_time, baseline, representation)
