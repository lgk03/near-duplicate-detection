import csv
import itertools
import json
import os
import numpy as np
import pandas as pd

base_path = '/Users/lgk/git/uni/web_test_generation/neural-embeddings-web-testing/'

APPS = ['addressbook', 'claroline', 'ppma', 'mrbs', 'mantisbt', 'dimeshift', 'pagekit', 'phoenix', 'petclinic']

OUTPUT_CSV = True # if True, write the results to a CSV file

setting = "within-apps" # within-apps or across-apps
print(f'====== Setting: {setting} ======')

filename = f'{base_path}0-Baselines_csv_results_table/rq2-ALT-{setting}.csv'
ss = pd.read_csv(f'{base_path}script/SS_threshold_set.csv')

DISTINCT_CLASS = 0
NEAR_DUP_CLASS = 1
print(f'NEAR_DUP_CLASS: {NEAR_DUP_CLASS} | DISTINCT_CLASS: {DISTINCT_CLASS}')

if __name__ == '__main__':
    os.chdir("..")

    if OUTPUT_CSV:
        if not os.path.exists(filename):
            header = ['Setting', 'App', 'Baseline', 'F1', 'Precision', 'Recall']
            with open(filename, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    for baseline in ['rted', 'pdiff']:

        for app in APPS:
            print(f'{app} - {baseline}')
            ss = ss[ss['appname'] == app]
            cluster_file_name = f'{base_path}output/{app}.json'

            pred_file = f'{base_path}script/distance_matrices/SS_as_distance_matrix_{setting}-{app}-{baseline}.csv'

            predictions = pd.read_csv(pred_file, index_col=0)

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
                        if model == []: # if model is empty, add the first state:
                            model.append(state)
                            bin_covered = True
                        else:
                            is_distinct = True
                            if bin_index == 0: # if the first state in the bin, i.e. the 'key' => compare with all existing states in the model
                                bin_index += 1
                                for ms in model: # for each state already in the model
                                    pred = predictions.loc[ms, state]
                                    if pred == NEAR_DUP_CLASS:
                                        is_distinct = False
                                        break

                                if is_distinct:
                                    model.append(state)
                                    bin_covered = True
                            else:
                                pred = predictions.loc[data[bin][bin_index - 1], state]
                                all_comparison_pairs.append((data[bin][bin_index - 1], state))
                                if pred == DISTINCT_CLASS:
                                    model.append(state)
                                    not_detected_near_duplicate_pairs.append((data[bin][bin_index - 1], state))
                                bin_index += 1

                    if bin_covered:
                        covered_bins.append(bin)

            unique_states_in_model = len(covered_bins)
            precision = unique_states_in_model / len(model) if len(model) > 0 else 0
            recall = len(covered_bins) / number_of_bins if number_of_bins > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            print(f"App: {app}, Baseline: {baseline}")
            print(f"Covered bins: {len(covered_bins)}")
            print(f"Number of bins: {number_of_bins}")
            print(f"Total number of states: {total_number_of_states}")
            print(f"Number of states in model: {len(model)}")
            print(f"Unique states in model: {unique_states_in_model}")
            print(f"Number of State-Pairs not detected as near-duplicates: {len(not_detected_near_duplicate_pairs)}")
            # print(f"State-Pairs not detected as near-duplicates: {not_detected_near_duplicate_pairs}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1_score:.4f}")

            if OUTPUT_CSV:
                with open(filename, 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow([setting, app, baseline, f1_score,precision, recall])
