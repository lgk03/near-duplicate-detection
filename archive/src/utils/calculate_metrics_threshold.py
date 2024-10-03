# functions to calculate the treshold for deciding which label should be predicted from a given similarity score

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score


# calculates the optimal threshold for the labels clone, near-duplicate and distinct + f1-score
# param file_path: path to the file containing the output of the pretrained model
def calculate_threshold(file_path):
    with open(file_path, 'r') as file:
        print(f'Evaluate HTML raw embeddings from {file_path.split("/")[-1]}:')
        lines = file.readlines()

    y_true = []
    scores = []

    for line in lines:
        parts = line.split(', ')
        human_classification = int(parts[4].split(':')[-1])
        cosine_similarity = float(parts[5].split(':')[-1])

        y_true.append(human_classification)
        scores.append(cosine_similarity)

    y_true = np.array(y_true)
    scores = np.array(scores)

    best_thresholds = [0, 0, 0]
    best_f1_scores = [0, 0, 0]

    for label in range(3):
        # Create binary labels for the current class vs. non-class
        y_true_class = (y_true == label).astype(int)

        # Define a range of thresholds to explore
        threshold_range = np.linspace(0, 1, 100)

        best_threshold = 0
        best_f1_score = 0

        # Evaluate performance for each threshold
        for threshold in threshold_range:
            predictions = (scores >= threshold).astype(int)

            # Calculate F1 score for the current label
            f1 = f1_score(y_true_class, predictions)

            # Check if the current threshold gives a better F1 score
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = threshold

        # Store the best threshold and F1 score for the current label
        best_thresholds[label] = best_threshold
        best_f1_scores[label] = best_f1_score

    # Print results
    for label in range(3):
        print(f'Best Threshold for Label {label}: {best_thresholds[label]}')
        print(f'Best F1 Score for Label {label}: {best_f1_scores[label]}')


if __name__ == '__main__':
    html_raw_path = ('trimmed_content_tags.txt')
    html_content_path = ('trimmed_content.txt')
    html_tags_path = ('trimmed_tags.txt')

    label_classification = {0: 'clone', 1: 'near-duplicate', 2: 'distinct'}

    calculate_threshold(html_raw_path)
