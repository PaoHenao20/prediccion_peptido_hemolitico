import os
import requests

def download_file_from_url(url: str, output_path: str):
    """
    Downloads a file from the specified URL and saves it to the given output path.

    Parameters:
    - url (str): Direct URL to the file (e.g., a CSV or FASTA file)
    - output_path (str): Local path where the file will be saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully: {output_path}")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")
