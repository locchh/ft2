'''
python utils/pull.py <repo_id>
'''
import os
import sys
from huggingface_hub import HfApi, hf_hub_download

# Initialize the API
api = HfApi()

# Define the repository (model) you want to download files from
model_id =  sys.argv[1] #"your-username/your-model-id"

# List all files in the repository
files = api.list_repo_files(model_id)

# Create a directory to save the files if it doesn't exist
# output_dir = "./model_files"
# os.makedirs(output_dir, exist_ok=True)

# Download each file
for file_name in files:
    print(f"Downloading {file_name}...")
    file_path = hf_hub_download(repo_id=model_id, filename=file_name)
    
    # Move downloaded file to the output directory
    # os.rename(file_path, os.path.join(output_dir, file_name))

print("All files downloaded successfully!")