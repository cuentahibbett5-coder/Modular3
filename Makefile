# Proyecto Modular 3 - CUCEI
# Simulación Monte Carlo de Aceleradores Lineales con Denoising mediante IA

.PHONY: help install run-tests train validate clean

help:
	@echo "Proyecto Modular 3 - Comandos disponibles:"
	@echo ""
	@echo "  make install          - Instalar dependencias"
	@echo "  make setup-env        - Crear entorno conda"
	@echo "  make run-tests        - Ejecutar tests unitarios"
	@echo "  make generate-data    - Generar dataset de entrenamiento"
	@echo "  make train            - Entrenar modelo MCDNet"
	@echo "  make validate         - Validar con gamma index"
	@echo "  make docs             - Compilar documentación LaTeX"
	@echo "  make clean            - Limpiar archivos temporales"

install:
	pip install -r requirements.txt

setup-env:
	conda env create -f environment.yml
	conda activate modular3

run-tests:
	python -m pytest tests/ -v

generate-data:
	python data/dataset_generator.py \
		--output data/training/ \
		--phantoms water bone lung \
		--fields 5 10 15 \
		--samples-per-config 20

train:
	python models/training.py \
		--data-dir data/training/ \
		--epochs 100 \
		--batch-size 4 \
		--lr 1e-4 \
		--device cuda

validate:
	python analysis/gamma_index.py \
		--reference data/validation/reference.mhd \
		--evaluated results/denoised.mhd \
		--criteria 3%/3mm \
		--output results/gamma/

docs:
	cd docs/latex && pdflatex main.tex && pdflatex main.tex

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf models/checkpoints/*.pth

workflow:
	bash scripts/run_complete_workflow.sh
