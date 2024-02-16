"""
this script checks if all the files in the Yandrapally dataset are following
the convention of: stateX.html, state3.html.content, state3.html.tags, state3.html.content_tags
"""
import os


def check_file_convention(base_name, folder_path):
    # Check if all expected extensions are present for the given base name
    expected_extensions = ['.html', '.html.content', '.html.tags', '.html.content_tags']
    missing_extensions = []

    for suffix in expected_extensions:
        expected_file = os.path.join(folder_path, base_name + suffix)
        if not os.path.exists(expected_file):
            missing_extensions.append(suffix)
            return missing_extensions

    return missing_extensions


def iterate_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"The specified folder '{folder_path}' does not exist.")
        return

    flag = False #indicates if the folder contains any files that do not follow the convention
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            base_name, ext = os.path.splitext(filename)
            if ext != '.html':
                continue
            missing_extensions = check_file_convention(base_name, folder_path)
            if len(missing_extensions) == 0:
                continue
            else:
                flag = True
                print(f"'{base_name}' is missing the following extensions: {missing_extensions}")
        elif os.path.isdir(file_path):
            print(f"Skipping subfolder: {file_path}")
        else:
            print(f"Skipping unknown file type: {file_path}")
    return flag


if __name__ == "__main__":
    base_target_folder = "GroundTruthModels-SS"

    for application_name in os.listdir(base_target_folder):
        application_folder = os.path.join(base_target_folder, application_name)
        if os.path.isdir(application_folder):
            print(f"\nProcessing application: {application_name}")
            if not iterate_folder(application_folder):
                print(f"All files in '{application_name}' follow the convention.")
        else:
            print(f"Skipping non-directory: {application_folder}")
