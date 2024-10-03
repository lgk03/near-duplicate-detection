import numpy as np
import pandas as pd
import os

# Load the data
df = pd.read_csv('SS.csv')

def trim_common_html(state1, state2):
    """
    Trims the common leading and trailing parts from two HTML page representations (content_tags. tags, content).

    :param state1: HTML content of the first page as a string.
    :param state2: HTML content of the second page as a string.
    :return: A tuple of the trimmed HTML contents.
    """
    # Find the common leading part
    leading_common_length = 0
    for x, y in zip(state1, state2):
        if x == y:
            leading_common_length += 1
        else:
            break

    # Find the common trailing part
    trailing_common_length = 0
    for x, y in zip(reversed(state1[leading_common_length:]), reversed(state2[leading_common_length:])):
        if x == y:
            trailing_common_length += 1
        else:
            break

    # Trim the common parts from both HTML contents
    trimmed_state1 = state1[leading_common_length: len(state1) - trailing_common_length]
    trimmed_state2 = state2[leading_common_length: len(state2) - trailing_common_length]
    if trimmed_state2 == "" or trimmed_state1 == "":
        trimmed_state1 = state1
        trimmed_state2 = state2

    return trimmed_state1, trimmed_state2


def fetch_file_content(file_path):
    """
    Reads the HTML file content from a given file path.
    :param file_path: Path to the HTML file.
    :return: Content of the HTML file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return ""

# analytics
total_trimmed_length = 0
total_original_length = 0
file_ending = "html.tags"
# html.content_tags
# html.content
# html.tags

# Iterate through the rows of the dataframe
for _, row in df.iterrows():
    # Construct the file paths for state1 and state2 HTML files
    state1_path = f"GroundTruthModels-SS/{row['appname']}/{row['state1']}.{file_ending}"
    state2_path = f"GroundTruthModels-SS/{row['appname']}/{row['state2']}.{file_ending}"

    # Fetch the HTML content for state1 and state2
    html1 = fetch_file_content(state1_path)
    html2 = fetch_file_content(state2_path)

    # Trim the common parts from the HTML content
    trimmed_html1, trimmed_html2 = trim_common_html(html1, html2)

    # Update the total lengths for average calculation
    total_original_length += len(html1) + len(html2)
    total_trimmed_length += len(trimmed_html1) + len(trimmed_html2)

# Calculate the average trimmed length
average_trimmed_length = total_trimmed_length / len(df) if len(df) > 0 else 0

# Calculate the average original length
average_original_length = total_original_length / len(df) if len(df) > 0 else 0

# Calculate the average percentage of content trimmed
average_percentage_trimmed = (100 -
            (average_trimmed_length / average_original_length * 100)) if average_original_length > 0 else 0

print("Average original length:", format(average_original_length, ".3f"))
print("Average trimmed length:", format(average_trimmed_length, ".3f"))
print("Average percentage trimmed:", format(average_percentage_trimmed, ".3f"))
