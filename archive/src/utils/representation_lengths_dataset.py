import os

directory_path = 'GroundTruthModels-SS'
repr_type = '.html.content'
repr_type_array = ['.html.content', '.html.tags', '.html.content_tags']

def calculate_average_chars_in_files(folder_path):
    total_chars = 0
    file_count = 0

    # List all files in the given folder
    for filename in os.listdir(folder_path):
        # Check if the file ends with '.html.content'
        if filename.endswith(repr_type):
            file_path = os.path.join(folder_path, filename)
            file_count += 1
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                total_chars += len(content)

    # Calculate average characters per file
    if file_count > 0:
        average_chars = total_chars / file_count
    else:
        average_chars = 0

    return average_chars

for repr_type in repr_type_array:
    total_average_chars = 0
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        # print(f"Analyzing '{folder_name}'...")
        if os.path.isdir(folder_path):
            average_chars = calculate_average_chars_in_files(folder_path)
            total_average_chars += average_chars
            # print(f"Average characters in files of '{folder_name}': {average_chars}")

    # Calculate the average characters per file for all folders
    if len(os.listdir(directory_path)) > 0:
        average_chars_all = total_average_chars / len(os.listdir(directory_path))
        print(f"Representation type - {repr_type} -> Avg # of characters: {average_chars_all}")