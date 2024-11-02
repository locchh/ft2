"""
python push_model.py your-username/your-model-id ./model_files true
"""
import os
import sys
from huggingface_hub import HfApi, create_repo, upload_file

def push_model_to_hub(repo_id, model_directory, private=True):
    # Initialize the API
    api = HfApi()
    
    # Create a repository (model) if it does not exist
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    # List all files in the model directory
    files = os.listdir(model_directory)

    # Upload each file
    for file_name in files:
        file_path = os.path.join(model_directory, file_name)
        print(f"Uploading {file_name}...")
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="model"  # Specify the type as a model
        )
    
    print("All files uploaded successfully!")

if __name__ == "__main__":
    # Command-line arguments
    repo_id = sys.argv[1]  # e.g., "your-username/your-model-id"
    model_directory = sys.argv[2]  # e.g., "./model_files"
    is_private = sys.argv[3].lower() == "true"  # Expect "true" or "false"
    
    push_model_to_hub(repo_id, model_directory, private=is_private)
