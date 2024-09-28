# Makefile for running ml pipeline

# Define variables
PYTHON = python
SRC_DIR = src
OUTPUT_DIR = output

# Default target
all: ml_pipeline

# Target to run ml_pipeline.py
ml_pipeline:
	$(PYTHON) $(SRC_DIR)/ml_pipeline.py

# Clean target to remove output files (if needed)
clean:
	rm -rf $(OUTPUT_DIR)/*

# Phony targets
.PHONY: all ml_pipeline clean
