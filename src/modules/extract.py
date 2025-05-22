import os
import requests

def download_file_from_url(url: str, output_path: str):
    """
    Descarga un archivo desde una URL y lo guarda en output_path.

    Parámetros:
    - url (str): URL directa al archivo (ej. CSV)
    - output_path (str): Ruta local donde se guardará el archivo
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Archivo descargado correctamente: {output_path}")
    else:
        raise Exception(f"Error al descargar archivo. Código de estado: {response.status_code}")
