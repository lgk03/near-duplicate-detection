import csv
import itertools
import json
import os
import numpy as np
import pandas as pd

base_path = '/Users/lgk/git/uni/web_test_generation/neural-embeddings-web-testing/'

APPS = ['addressbook', 'claroline', 'ppma', 'mrbs', 'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic']

OUTPUT_CSV = True # if True, write the results to a CSV file
ADJUSTED_CW = True # if True, use the trained models with the adjusted class weights

setting = "within_apps" # within_apps or across_apps
print(f'====== Setting: {setting} ======')

filename = f'{base_path}0-BERT-SAF_csv_results_table/{"CWAdj-" if ADJUSTED_CW else ""}rq2-ALT-{setting}.csv'

DISTINCT_CLASS = 0
NEAR_DUP_CLASS = 1
print(f'NEAR_DUP_CLASS: {NEAR_DUP_CLASS} | DISTINCT_CLASS: {DISTINCT_CLASS}')

if __name__ == '__main__':
    os.chdir("..")

    if OUTPUT_CSV:
        if not os.path.exists(filename):
            header = ['Setting', 'App', 'Feature', 'F1', 'Precision', 'Recall']
            with open(filename, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    for feature in ['content_tags', 'content', 'tags']:
        prediction_column = f'{feature}-PREDICTION' # 'HUMAN_CLASSIFICATION'
        if setting=='across_apps' and ADJUSTED_CW and feature not in ['content']:
            print(f"Skipping {feature} for adjusted class weights model variant | not trained yet.")
            continue

        for app in APPS:
            print(f'{app} - {feature} - {prediction_column} - {setting}')
            cluster_file_name = f'output/{app}.json' # file listing all clusters for the app and the corresponding states
            pred_file = f'model_predictions_ss/{setting}/{"CWAdj-" if ADJUSTED_CW else ""}{app}.csv' # ss with predictions by the model
            predictions = pd.read_csv(pred_file)

            model = [] # list of states that are included in model
            covered_bins = []
            number_of_bins = 0
            total_number_of_states = 0
            not_detected_near_duplicate_pairs = []
            all_comparison_pairs = []

            with open(cluster_file_name, 'r') as f:
                data = json.load(f)
                for bin in data:
                    bin_index = 0 # index of the current state in the bin
                    number_of_bins += 1
                    bin_covered = False
                    for state in data[bin]:
                        total_number_of_states += 1
                        if model == []: # if model is empty, add the first state
                            model.append(state)
                            bin_covered = True
                        else:
                            is_distinct = True
                            if bin_index == 0: # if the first state in the bin, i.e. the 'key' => compare with all existing states in the model
                                bin_index += 1
                                for ms in model: # for each state already in the model
                                    matching_row = predictions.loc[(predictions['state1'] == ms) & (predictions['state2'] == state)]
                                    if matching_row.empty:
                                        matching_row = predictions.loc[(predictions['state1'] == state) & (predictions['state2'] == ms)]

                                    if not matching_row.empty:
                                        all_comparison_pairs.append((ms, state))
                                        if matching_row[prediction_column].values[0] == NEAR_DUP_CLASS: # current state is clone/ND to a state in the model => do not add to model
                                            is_distinct = False
                                            break
                                    else:
                                        print(f"Missing: {ms} - {state}")

                                if is_distinct:
                                    model.append(state)
                                    bin_covered = True

                            else: # if not the first state in the bin,  => compare with the previous state in the bin
                                matching_row = predictions.loc[(predictions['state1'] == data[bin][bin_index - 1]) & (predictions['state2'] == state)]
                                if matching_row.empty:
                                    matching_row = predictions.loc[(predictions['state1'] == state) & (predictions['state2'] == data[bin][bin_index - 1])]
                                if not matching_row.empty:
                                    all_comparison_pairs.append((data[bin][bin_index - 1], state))
                                    if matching_row[prediction_column].values[0] == DISTINCT_CLASS:
                                        model.append(state) # add the cloned state to the model as it was not classified as a clone
                                        not_detected_near_duplicate_pairs.append((data[bin][bin_index - 1], state))
                                else:
                                    print(f"Missing: {data[bin][bin_index - 1]} - {state}")
                                bin_index += 1

                    if bin_covered:
                        covered_bins.append(bin)

            # Calculate Precision, Recall, F1 Score
            unique_states_in_model = len(covered_bins)
            precision = unique_states_in_model / len(model) if len(model) > 0 else 0
            recall = len(covered_bins) / number_of_bins if number_of_bins > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            print(f"App: {app}")
            print(f"Covered bins: {len(covered_bins)}")
            print(f"Number of bins: {number_of_bins}")
            print(f"Total number of states: {total_number_of_states}")
            print(f"Number of states in model: {len(model)}")
            print(f"Unique states in model: {unique_states_in_model}")
            print(f"Number of State-Pairs not detected as near-duplicates: {len(not_detected_near_duplicate_pairs)}")
            # print(f"State-Pairs not detected as near-duplicates: {not_detected_near_duplicate_pairs}")
            # print(f"Number of State-Pairs compared: {len(all_comparison_pairs)}")
            # print(f"State-Pairs compared: {all_comparison_pairs}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1_score:.4f}")

            # Write results to CSV if needed
            if OUTPUT_CSV:
                with open(filename, 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow([setting, app, feature, f1_score, precision, recall])
