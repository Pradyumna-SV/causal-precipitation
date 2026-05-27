# Reproducible pipeline (requires .venv + ~/.cdsapirc or project .cdsapirc)
PYTHON ?= .venv/bin/python

.PHONY: preprocess discovery inference figures robustness holdout diagnostics pipeline verify manuscript-brief pooled-ace search-grid experiment-matrix

preprocess:
	$(PYTHON) scripts/02_preprocess.py

discovery:
	$(PYTHON) scripts/03_causal_discovery.py

inference:
	$(PYTHON) scripts/04_causal_inference.py

figures:
	$(PYTHON) scripts/05_figures.py

robustness:
	$(PYTHON) scripts/robustness_bundle.py

holdout:
	$(PYTHON) scripts/holdout_validation.py

diagnostics:
	$(PYTHON) scripts/feedback_diagnostics.py

pooled-ace:
	$(PYTHON) scripts/pooled_ace.py

search-grid:
	$(PYTHON) scripts/run_search_grid.py

# Full estimand/outcome/aggregation grid + log (see config/experiments_matrix.yaml).
experiment-matrix:
	$(PYTHON) scripts/run_experiment_matrix.py

pipeline: preprocess discovery inference figures robustness holdout pooled-ace diagnostics verify manuscript-brief

verify:
	$(PYTHON) scripts/verify_pipeline.py

# CCAI-oriented claim bundle from results/ (run after verify PASS).
manuscript-brief:
	$(PYTHON) scripts/manuscript_brief.py
