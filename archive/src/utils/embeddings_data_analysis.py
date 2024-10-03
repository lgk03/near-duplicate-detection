import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

trimmed_content_tags_path = ('trimmed_content_tags.txt')
trimmed_content_path = ('trimmed_content.txt')
trimmed_tags_path = ('trimmed_tags.txt')
whole_content_tags_path = ('whole_content_tags.txt')
whole_content_path = ('whole_content.txt')
whole_tags_path = ('whole_tags.txt')


def get_df_from_embeddings_file(path, application_name=None):
    """
        Function to get a dataframe from an embeddings file.

        Parameters:
        path (str): Path to the embeddings file, .txt
        application_name (str): Name of one of the applications of the nine web-apps in the dataset, e.g. 'addressbook', 'ppma', ...

        Returns:
        pandas.DataFrame: A dataframe with the columns 'index', 'application', 'state1', 'state2', 'clone/near-duplicate', 'cosine_similarity'
    """
    indices = []
    apps = []
    state1 = []
    state2 = []
    human_classifications = []
    cosine_similarities = []
    with open(path, 'r') as file:
        for line in file:
            parts = [part.strip() for part in line.split(',')]
            if application_name is not None:
                if application_name != parts[1]:
                    continue
            indices.append(int(parts[0].strip('()')))
            apps.append(parts[1])
            state1.append(parts[2].split(':')[1])
            state2.append(parts[3].split(':')[1])
            # => binary classification, 1 = clone/near-dup, 0 = distinct
            label = int(parts[4].split(':')[1])
            if label == 2:
                label = 0
            elif label == 1 or label == 0:
                label = 1
            human_classifications.append(label)
            cosine_similarities.append(float(parts[5].split(':')[1]))

    df = pd.DataFrame({
        'index': indices,
        'application': apps,
        'state1': state1,
        'state2': state2,
        'clone/near-duplicate': human_classifications,
        'cosine_similarity': cosine_similarities
    })

    # sanity check
    if application_name and df['application'].nunique() != 1:
        print('ERROR: Application name is not unique!')

    return df


def get_optimal_threshold_for_application(df, application_name=None):
    """
        Function to infer the optimal threshold using the ROC curve.

        Parameters:
        df (pandas.DataFrame): The dataframe to analyze.
        application_name (str): Name of one of the applications of the nine web-apps in the dataset, e.g. 'addressbook', 'ppma', ...

        Returns:
        float: The optimal threshold for the given application.
    """
    fpr, tpr, thresholds = roc_curve(df['clone/near-duplicate'], df['cosine_similarity'])
    optimal_idx = np.argmax(tpr - fpr)
    opt_threshold = thresholds[optimal_idx]
    return opt_threshold


# Function to classify based on the threshold
def classify(cosine_similarity, threshold):
    return 1 if cosine_similarity >= threshold else 0


def predict_and_evaluate(df, optimal_threshold):
    """
        Function to predict and evaluate the results.

        Parameters:
        df (pandas.DataFrame): The dataframe to analyze.
        optimal_threshold (float): The optimal threshold to use for classification.

        Returns:
        tuple: The F1 score and accuracy of the predictions.
    """
    # Apply the classification function to the cosine_similarity column
    df['predicted'] = df['cosine_similarity'].apply(lambda x: classify(x, optimal_threshold))

    # Calculate metrics
    precision = precision_score(df['clone/near-duplicate'], df['predicted'])
    recall = recall_score(df['clone/near-duplicate'], df['predicted'])
    f1 = f1_score(df['clone/near-duplicate'], df['predicted'])
    accuracy = accuracy_score(df['clone/near-duplicate'], df['predicted'])

    # # Print metrics
    # print('Precision: %f' % precision)
    # print('Recall: %f' % recall)
    # print('F1 Score: %f' % f1)
    # print('Accuracy: %f' % accuracy)

    return f1, accuracy


if __name__ == '__main__':
    applications = ['addressbook', 'claroline', 'dimeshift', 'mantisbt', 'mrbs', 'pagekit', 'petclinic', 'phoenix',
                    'ppma']
    embeddings_files = [trimmed_content_tags_path, trimmed_content_path, trimmed_tags_path, whole_content_tags_path, whole_content_path, whole_tags_path]

    for embeddings_file in embeddings_files:
        avg_f1 = 0
        avg_accuracy = 0
        print('Embeddings file: ' + embeddings_file.split('/')[-1].split('.')[0])
        df = get_df_from_embeddings_file(embeddings_file)
        optimal_threshold = get_optimal_threshold_for_application(df)
        all_f1, all_accuracy = predict_and_evaluate(df, optimal_threshold)
        print('All F1: ', all_f1, ', all acc: ', all_accuracy)
        for application in applications:
            df = get_df_from_embeddings_file(embeddings_file, application)
            optimal_threshold = get_optimal_threshold_for_application(df, application)
            app_f1, app_accuracy = predict_and_evaluate(df, optimal_threshold)
            avg_f1 += app_f1
            avg_accuracy += app_accuracy
        avg_f1 /= len(applications)
        avg_accuracy /= len(applications)
        print('avg F1: ', avg_f1, ', avg acc: ', avg_accuracy)
        print('-----------------------------------------------')
