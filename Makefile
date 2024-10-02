# Main Makefile to orchestrate the entire pipeline

# Phony targets
.PHONY: all clean ingest ml

# Default target
all: ingest ml

# Target to run data ingestion
ingest:
	@echo "Running data ingestion..."
	$(MAKE) -C x01_ingest_data

# Target to run ML pipeline
ml: ingest
	@echo "Running ML pipeline..."
	$(MAKE) -C x02_ml_pipeline

# Target to clean all subdirectories
clean:
	@echo "Cleaning all subdirectories..."
	$(MAKE) -C x01_ingest_data clean
	$(MAKE) -C x02_ml_pipeline clean

# Help target
help:
	@echo "Available targets:"
	@echo "  all       : Run the entire pipeline (default)"
	@echo "  ingest    : Run data ingestion"
	@echo "  ml        : Run ML pipeline (depends on ingest)"
	@echo "  clean     : Clean all subdirectories"
	@echo "  help      : Show this help message"
