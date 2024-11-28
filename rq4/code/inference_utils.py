from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

# dataset base path, should included folders for each app and in each app folder the states in different representations (.html, .content, .tags, .content_tags)
dataset_base = '/Users/lgk/Documents/uni/BA-Local/Data/WebEmbed-97k-state-pairs'

# dataset related
def load_state(appname, state, representation, baseline='bert-base'):
    """
    Load the HTML content of a state from the dataset folder.
    
    Args:
    appname (str): The name of the app.
    state (str): The name of the state.
    representation (str): The representation of the state.
    
    Returns:
    str: The HTML content of the state in the requested representation.
    """
    try:
        with open(f"{dataset_base}/{appname}/{state}.html.{representation}", "r") as f:
            if baseline == 'webembed':
                return json.load(f)
            return ' '.join(json.load(f))
    except Exception as e:
        print(f"An error occurred while loading the state: {e}")
        return None
    
def print_report(app_time, baseline, representation):
    """
    Print the results of the inference time experiment.
    """
    print(f"Results for {baseline} using {representation} representation:")
    for app, time in app_time.items():
        print(f"{app}: {time:.2f} seconds")
    print(f"Avg. time: {sum(app_time.values()) / len(app_time):.2f} seconds")


# everything transformer related
def load_model_and_tokenizer(model_path):
    """
    Load a model and tokenizer from the Hugging Face Hub.
    
    Args:
    model_path (str): The path of the model on the Hugging Face Hub
    
    Returns:
    tuple: (model, tokenizer)
    """
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set the model to evaluation mode
        model.eval()
        
        return model, tokenizer
    
    except Exception as e:
        print(f"An error occurred while loading the model and tokenizer: {e}")
        return None, None
    
def trim_common_html(state1:str, state2:str) -> tuple[str, str]:
    """
    Trims the common leading and trailing parts from two HTML page representations (content_tags, tags, content).

    :param state1: HTML content of the first page as a string.
    :param state2: HTML content of the second page as a string.
    :return: A tuple of the trimmed HTML contents.
    """
    leading_common_length = 0
    for x, y in zip(state1, state2):
        if x == y:
            leading_common_length += 1
        else:
            break
    trailing_common_length = 0
    for x, y in zip(reversed(state1[leading_common_length:]), reversed(state2[leading_common_length:])):
        if x == y:
            trailing_common_length += 1
        else:
            break
    trimmed_state1 = str(state1[leading_common_length: len(state1) - trailing_common_length])
    trimmed_state2 = str(state2[leading_common_length: len(state2) - trailing_common_length])
    if trimmed_state1.startswith('\",'): trimmed_state1 = trimmed_state1[3:]
    if trimmed_state2.startswith('\",'): trimmed_state2 = trimmed_state2[3:]
    if trimmed_state1.endswith('\"'): trimmed_state1 = trimmed_state1[:-3]
    if trimmed_state2.endswith('\"'): trimmed_state2 = trimmed_state2[:-3]

    # if one page is subset of the other page
    if trimmed_state2 == "" or trimmed_state1 == "":
      return state1, state2
    
    return trimmed_state1, trimmed_state2


def preprocess_for_inference(state1, state2, tokenizer):
    trimmed_state1, trimmed_state2 = trim_common_html(state1, state2)
    tokenized_inputs = tokenizer(trimmed_state1, trimmed_state2,
                                 padding='max_length',
                                 truncation='longest_first',
                                 max_length=512,
                                 return_tensors='pt')
    return tokenized_inputs


def get_prediction(model, inputs):
    with torch.no_grad():  # disable gradient computation
        outputs = model(**inputs)
    # extract logits and apply softmax
    probabilities = torch.softmax(outputs.logits, dim=-1)
    # predict the class with the highest probability
    predicted_class_id = probabilities.argmax(dim=-1).item()
    return predicted_class_id


# everything WebEmbed related
# TODO: implement