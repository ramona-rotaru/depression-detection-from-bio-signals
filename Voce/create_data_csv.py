import numpy as np  
import pandas as pd
import os

# Function to process each CSV file
def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if both 'Participant_ID' and 'PHQ8_Binary' columns are present
    if 'Participant_ID' in df.columns and 'PHQ8_Binary' in df.columns:
        # Extract Participant_ID and PHQ8_Binary columns
        participant_id = df['Participant_ID']
        phq8_binary = df['PHQ8_Binary']
        
        # Create a DataFrame with only Participant_ID and PHQ8_Binary columns
        new_df = pd.DataFrame({'Participant_ID': participant_id, 'PHQ8_Binary': phq8_binary})
        
        return new_df
    else:
        print(f"Skipping {file_path} because one or both of the required columns are missing.")
        return None

# Path to the folder containing the CSV files
folder_path = 'E:/Daicwoz/'

# List to store DataFrames from each CSV file
dfs = []

# Iterate through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # Process the CSV file
        df = process_csv(file_path)
        if df is not None:
            dfs.append(df)

# Check if any CSV files were processed
if dfs:
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    
     # Sort the combined DataFrame by Participant_ID
    combined_df.sort_values(by='Participant_ID', inplace=True)

    # Write the combined DataFrame to a new CSV file
    combined_df.to_csv('combined_data.csv', index=False)
else:
    print("No valid CSV files found.")
