import os
import requests

def get_github_contents(repo, path=""):
    """
    Fetch the contents of a directory in a GitHub repository.
    """
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

def download_file(url, local_path):
    """
    Download a file from a URL to a local path.
    """
    response = requests.get(url)
    response.raise_for_status()
    with open(local_path, 'wb') as f:
        f.write(response.content)

def main():
    repo = "AlexanderAivazidis/cell2fate_notebooks"
    base_path = ""
    base_url = "https://raw.githubusercontent.com/"
    contents = get_github_contents(repo, base_path)

    for content in contents:
        # Check if content is a file and ends with .ipynb
        if content['type'] == 'file' and content['name'].endswith('.ipynb'):
            notebook_path = content['path']
            notebook_url = f"{base_url}{repo}/main/{notebook_path}"
            local_path = os.path.join(base_path, os.path.basename(notebook_path))
            # Ensure the local directory exists
            local_dir = os.path.dirname(local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            
            print(f"Downloading {notebook_url} to {local_path}")
            download_file(notebook_url, local_path)
        elif content['type'] == 'dir':
            # Recursively download notebooks in subdirectories
            sub_contents = get_github_contents(repo, content['path'])
            for sub_content in sub_contents:
                if sub_content['type'] == 'file' and sub_content['name'].endswith('.ipynb'):
                    notebook_path = sub_content['path']
                    notebook_url = f"{base_url}{repo}/main/{notebook_path}"
                    local_path = "./notebooks/"+notebook_path
                    
                    # Ensure the local directory exists
                    local_dir = os.path.dirname(local_path)
                    if not os.path.exists(local_dir):
                        os.makedirs(local_dir)

                    print(f"Downloading {notebook_url} to {local_path}")
                    download_file(notebook_url, local_path)

if __name__ == "__main__":
    main()
