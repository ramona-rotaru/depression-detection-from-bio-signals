import zipfile
import os
import shutil

# Path to your folder containing zipped subfolders
folder_path = "E:/Daicwoz/"

def extract_and_move_wav_files(zip_file_path):
    # Create a directory to move .wav files into
    output_dir = os.path.join(os.path.dirname(zip_file_path), "extracted_wav_files")
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # List all files in the zip archive
        file_list = zip_ref.namelist()
        # Filter out .wav files
        wav_files = [file for file in file_list if file.endswith('.wav')]
        # Extract each .wav file and move it to the output directory
        for wav_file in wav_files:
            # Check if the .wav file already exists in the output directory
            if not os.path.exists(os.path.join(output_dir, os.path.basename(wav_file))):
                # Extract the file contents into memory (without extracting to disk)
                with zip_ref.open(wav_file) as file:
                    # Read the content of the .wav file
                    wav_content = file.read()
                    # Write the content to a new .wav file in the output directory
                    output_file_path = os.path.join(output_dir, os.path.basename(wav_file))
                    with open(output_file_path, 'wb') as out_file:
                        out_file.write(wav_content)

# Iterate through all zip files in the folder and extract .wav files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.zip'):
            zip_file_path = os.path.join(root, file)
            print(zip_file_path)
            extract_and_move_wav_files(zip_file_path)

# Function to extract .wav files from zip archives and move them to a new directory
def extract_and_move_wav_files(zip_file_path):
    # Create a directory to move .wav files into
    output_dir = os.path.join(os.path.dirname(zip_file_path), "extracted_wav_files")
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # List all files in the zip archive
        file_list = zip_ref.namelist()
        # Filter out .wav files
        wav_files = [file for file in file_list if file.endswith('.wav')]
        # Extract each .wav file and move it to the output directory
        for wav_file in wav_files:
            # Extract the file contents into memory (without extracting to disk)
            with zip_ref.open(wav_file) as file:
                # Read the content of the .wav file
                wav_content = file.read()
                # Write the content to a new .wav file in the output directory
                output_file_path = os.path.join(output_dir, os.path.basename(wav_file))
                with open(output_file_path, 'wb') as out_file:
                    out_file.write(wav_content)

# Iterate through all zip files in the folder and extract .wav files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.zip'):
            zip_file_path = os.path.join(root, file)
            extract_and_move_wav_files(zip_file_path)
