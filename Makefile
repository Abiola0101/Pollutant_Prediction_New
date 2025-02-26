# Project Variables
VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# Default arguments (empty)
ARGS=

# Create virtual environment
init:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete!"
	@echo "Run 'source $(VENV)/bin/activate' (Linux/macOS) or '.venv\Scripts\activate' (Windows) to activate."

# Create virtual environment for CPU
init-cpu:
	@echo "Creating virtual environment for CPU..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies..."	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete!"
	@echo "Run 'source $(VENV)/bin/activate' (Linux/macOS) or '.venv\Scripts\activate' (Windows) to activate."


# Install dependencies
install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

# Run the data loading script
load_data:
	@echo " Running load_data.py..."
	$(PYTHON) src/load_data.py

# Run the preprocessing script
preprocess_data:
	@echo " Running preprocess.py..."
	$(PYTHON) src/preprocess.py

# Run feature engineering
feature_engineering:
	@echo "Running feature_engineering.py..."
	$(PYTHON) src/feature_engineering.py

# Run training script
train:
	@echo "Running train.py..."
	$(PYTHON) src/train.py

# Run prediction script
predict:
	@echo "Running predict.py..."
	$(PYTHON) src/predict.py

# Run evaluation script
evaluate:
	@echo "Running evaluate.py..."
	$(PYTHON) src/evaluate.py

# Run the full pipeline
run:
	@echo "Running main.py..."
	$(PYTHON) main.py

# Clean project: remove cache and old virtual environment
clean:
	@echo "Cleaning project..."
	rm -rf __pycache__ .venv
	find . -type d -name "__pycache__" -exec rm -r {} +
	@echo "Cleanup complete!"

# Show available commands
help:
	@echo " Makefile commands:"
	@echo "  make init            - Create virtual environment and install dependencies"
	@echo "  make init-cpu        - Create virtual environment for CPU and install dependencies"
	@echo "  make init-gpu        - Create virtual environment for GPU and install dependencies"
	@echo "  make install         - Install dependencies from requirements.txt"
	@echo "  make load_data       - Run the load_data script"
	@echo "  make preprocess_data - Run the preprocess script"
	@echo "  make feature_engineering - Run the feature engineering script"
	@echo "  make train           - Train the model"
	@echo "  make predict         - Run the prediction script"
	@echo "  make evaluate        - Evaluate the model"
	@echo "  make run             - Run the full ML pipeline (main.py)"
	@echo "  make clean           - Remove cache files and virtual environment"