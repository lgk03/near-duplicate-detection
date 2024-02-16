# script to evaluate the output of feeding raw html data to a pretrained model using the labeled data
from collections import Counter
import embeddings_data_analysis as eda

def evaluate_cosine_sim(file_path):
    # arrrays of the form: [similarity, index]
    highest_cosine_similarity_on_clone = [0, -1]
    lowest_cosine_similarity_on_clone = [1, -1]

    highest_cosine_similarity_on_near_dup = [0, -1]
    lowest_cosine_similarity_on_near_dup = [1, -1]

    highest_cosine_similarity_distinct = [0, -1]
    lowest_cosine_similarity_distinct = [1, -1]

    with open(file_path, 'r') as f:
        # (0), addressbook, State1:index, State2:state3, Human Classification:2, cosine similarity: 0.8541666269302368
        for line in f:
            split = line.split(',')
            index = split[0]
            application_name = split[1]
            state1 = split[2].split(':')[1]
            state2 = split[3].split(':')[1]
            human_classification = int(split[4].split(':')[1])
            cosine_similarity = float(split[5].split(':')[1])

            if human_classification == 0:
                if highest_cosine_similarity_on_clone[0] < cosine_similarity < 1:
                    highest_cosine_similarity_on_clone = [cosine_similarity, index]
                if 0 < cosine_similarity < lowest_cosine_similarity_on_clone[0]:
                    lowest_cosine_similarity_on_clone = [cosine_similarity, index]

            elif human_classification == 1:
                if highest_cosine_similarity_on_near_dup[0] < cosine_similarity < 1:
                    highest_cosine_similarity_on_near_dup = [cosine_similarity, index]
                if 0 < cosine_similarity < lowest_cosine_similarity_on_near_dup[0]:
                    lowest_cosine_similarity_on_near_dup = [cosine_similarity, index]

            elif human_classification == 2:
                if highest_cosine_similarity_distinct[0] < cosine_similarity < 1:
                    highest_cosine_similarity_distinct = [cosine_similarity, index]
                if 0 < cosine_similarity < lowest_cosine_similarity_distinct[0]:
                    lowest_cosine_similarity_distinct = [cosine_similarity, index]

        print('highest_cosine_similarity_on_clone: ', highest_cosine_similarity_on_clone)
        print('lowest_cosine_similarity_on_clone: ', lowest_cosine_similarity_on_clone)
        print('\n')
        print('highest_cosine_similarity_on_near_dup: ', highest_cosine_similarity_on_near_dup)
        print('lowest_cosine_similarity_on_near_dup: ', lowest_cosine_similarity_on_near_dup)
        print('\n')
        print('highest_cosine_similarity_distinct: ', highest_cosine_similarity_distinct)
        print('lowest_cosine_similarity_distinct: ', lowest_cosine_similarity_distinct)


# function to check whether the 2 outputs are consistent in terms of indices line per line
def check_consistency_of_outputs(first_file_path, second_file_path):
    with open(first_file_path, 'r') as f1:
        with open(second_file_path, 'r') as f2:
            for line1, line2 in zip(f1, f2):
                print(line1)
                print(line2)
                index1 = line1.split(',')[0]
                index2 = line2.split(',')[0]
                state1_1 = line1.split(',')[2].split(':')[1]
                state1_2 = line2.split(',')[2].split(':')[1]
                state2_1 = line1.split(',')[3].split(':')[1]
                state2_2 = line2.split(',')[3].split(':')[1]

                if state1_1 != state1_2:
                    print(f'{index1} vs {index2} Inconsistent states: {state1_1} vs {state1_2}')
                    return False
                if state2_1 != state2_2:
                    print(f'{index1} vs {index2} Inconsistent states: {state2_1} vs {state2_2}')
                    return False

                if index1 != index2:
                    print(f'Inconsistent indices: {index1} vs {index2}')
                    return False
    return True


# helper function to calculate the prediction of a label given a line of output (i.e. '(0), addressbook,
# State1:index, State2:state3, Human Classification:2, cosine similarity: 0.9661457538604736')
# returns [prediction, human_classification]
def get_prediction_humanclassification_given_threshold(line, threshold_0, threshold_1, threshold_2):
    parts = line.split(', ')
    human_classification = int(parts[4].split(':')[-1])
    cosine_similarity = float(parts[5].split(':')[-1])
    if cosine_similarity > threshold_0:
        return [0, human_classification]
    elif threshold_1 <= cosine_similarity <= threshold_2:
        return [1, human_classification]
    else:
        return [2, human_classification]

# calculates the Classification Rate given several classification approaches
# param predictions: list of lists of predictions
# param truth: list of the human classifications
# returns the final Classification Error Rate
def majority_prediction(predictions, truth):
    if not all(len(sublist) == len(predictions[0]) for sublist in predictions): print("Inconsistent sizes"); return -1
    transposed_lists = zip(*predictions)

    majority_predictions = []
    # Iterate through the transposed lists
    for index_values in transposed_lists:
        # Use Counter to find the majority prediction at the current index
        counter = Counter(index_values)
        majority_pred = counter.most_common(1)[0][0]
        majority_predictions.append(majority_pred)

    error_count = 0
    for i in range(len(majority_predictions)):
        if majority_predictions[i] != truth[i]:
            error_count += 1
    return error_count/len(majority_predictions)


# function to classify the output of the pretrained model into clone, near-duplicate and distinct
# return the accuracy for each label [accuracy_clone(0), accuracy_near_dup(1), accuracy_distinct(2), prediction_list]
def evaluate_thresholds_on_data(file_path, threshold_0, threshold_1, threshold_2):
    # (0), addressbook, State1:index, State2:state3, Human Classification:2, cosine similarity: 0.8541666269302368
    prediction_list = []
    truth_list = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    correct_predictions_0 = 0
    correct_predictions_1 = 0
    correct_predictions_2 = 0

    total_0 = 0
    total_1 = 0
    total_2 = 0

    classification_errors = 0

    # Iterate through each line in the file
    for line in lines:
        arr = get_prediction_humanclassification_given_threshold(line, threshold_0, threshold_1, threshold_2)
        prediction = arr[0]
        prediction_list.append(prediction)
        human_classification = arr[1]
        truth_list.append(human_classification)

        if human_classification == 0:
            total_0 += 1
        elif human_classification == 1:
            total_1 += 1
        elif human_classification == 2:
            total_2 += 1
        # Check if the prediction is correct for each label
        if prediction == human_classification == 0:
            correct_predictions_0 += 1
        elif prediction == human_classification == 1:
            correct_predictions_1 += 1
        elif prediction == human_classification == 2:
            correct_predictions_2 += 1
        else:
            classification_errors += 1

    # Calculate accuracy for each label
    total_entries = len(lines)
    accuracy_0 = correct_predictions_0 / total_0
    accuracy_1 = correct_predictions_1 / total_1
    accuracy_2 = correct_predictions_2 / total_2

    print(f"Classification error rate: {classification_errors / total_entries}")
    print(f"Accuracy for Clone: {accuracy_0}")
    print(f"Accuracy for Near-Duplicate: {accuracy_1}")
    print(f"Accuracy for Distinct: {accuracy_2}")
    return [accuracy_0, accuracy_1, accuracy_2, prediction_list, truth_list]



if __name__ == '__main__':

    # sanity checks, ensure that the outputs are consistent (always considering the same state pairs
    if not check_consistency_of_outputs(eda.whole_content_tags_path, eda.whole_content_path):
        #or not check_consistency_of_outputs(eda.whole_content_tags_path, eda.whole_tags_path) or not check_consistency_of_outputs(eda.whole_content_path, eda.whole_tags_path):
        print("Sanity checks not passed, exit")
        exit()
    print("passed")

    # get information about outliers:
    # evaluate_cosine_sim(path_output_html)
    # evaluate_cosine_sim(path_output_html_content)
    # evaluate_cosine_sim(path_output_html_tags)

    # print('Evaluate .html raw embeddings:')
    # thresholds_html = [0.98998998998999, 0.8908908908908909, 0.0]
    # accuracies_html = evaluate_thresholds_on_data(path_output_html, *thresholds_html)
    #
    # print('\nEvaluate .html.content raw embeddings:')
    # thresholds_html_content = [0.9769769769769769, 0.8238238238238238, 0.0]
    # accuracies_html_content = evaluate_thresholds_on_data(path_output_html_content, *thresholds_html_content)
    #
    # print('\nEvaluate .html.tags raw embeddings:')
    # thresholds_html_tags = [0.98989898989899, 0.888888888888889, 0.0]
    # accuracies_html_tags = evaluate_thresholds_on_data(path_output_html_tags, *thresholds_html_tags)
    #
    # prediction_list_html = accuracies_html[3]
    # prediction_list_html_content = accuracies_html_content[3]
    # prediction_list_html_tags = accuracies_html_tags[3]
    #
    # print(f"Classification Error Rate: {majority_prediction([prediction_list_html, prediction_list_html_content, prediction_list_html_tags], accuracies_html[4])}")
