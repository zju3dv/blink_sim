import os
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path
import shutil
import zipfile

def download_dataset(repo_id, branch, output_dir):
    """
    Download all files from a Hugging Face dataset repository
    """
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize the Hugging Face API
        api = HfApi()

        # Get the list of files in the repository
        print(f"Fetching file list from {repo_id}...")
        files = api.list_repo_files(repo_id, revision=branch, repo_type="dataset")

        # Download each file
        for file in files:
            try:
                print(f"Downloading: {file}")
                # Skip .gitattributes and other hidden files if desired
                if file.startswith('.'):
                    continue
                    
                # Download the file
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    revision=branch,
                    repo_type="dataset"
                )
                
                # Handle zip files
                if file.endswith('.zip'):
                    print(f"Extracting zip file: {file}")
                    with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                else:
                    # Create necessary subdirectories
                    target_path = output_dir / file
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy the file to the target location
                    shutil.copy2(downloaded_path, target_path)
                
            except Exception as e:
                print(f"Error downloading {file}: {str(e)}")
                continue

        print(f"\nDownload complete! Files saved to: {output_dir}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":

    repo_id = "BlinkVision/BlinkSim_Assets"
    branch = "main"
    output_dir = "./data"
    
    download_dataset(repo_id, branch, output_dir)
