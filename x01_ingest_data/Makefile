# Makefile for ingest_data

# Variables
PYTHON = python
SRC_DIR = src
CONFIG_DIR = config
INPUT_DIR = input
OUTPUT_DIR = output

# Default target
all: ingest_data

# Create output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Run the ingest_data.py script
ingest_data: $(OUTPUT_DIR)
	$(PYTHON) $(SRC_DIR)/ingest_data.py

# Clean the output directory
clean:
	rm -rf $(OUTPUT_DIR)/*

# Phony targets
.PHONY: all ingest_data clean
