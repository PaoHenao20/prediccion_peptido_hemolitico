# Nombre del entorno
ENV_NAME=prediccion_hemolitico
CONDA_ACTIVATE = source $(HOME)/miniconda3/etc/profile.d/conda.sh && conda activate $(ENV_NAME)

run:
	$(CONDA_ACTIVATE) && python src/modules/main.py

install:
	$(CONDA_ACTIVATE) && pip install -r requirements.txt

jupyter:
	$(CONDA_ACTIVATE) && jupyter notebook

clean:
	rm -rf __pycache__ *.pyc .pytest_cache
