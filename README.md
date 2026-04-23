# causal-precipitation

## Project

This project applies causal discovery and causal inference methods to ERA5 climate reanalysis data to identify and quantify the causal drivers of extreme precipitation over the western United States. We use PCMCI+ for time-series causal discovery and inverse probability weighting / regression adjustment for estimating average causal effects.

---

## Team

| Member | Owned Script | Primary Environment |
|--------|-------------|---------------------|
| Pradyumna | `01_download_era5.py` | Nautilus HPC |
| Suditi | `02_preprocess.py` | Local (macOS/Linux) |
| Anirudh | `03_causal_discovery.py` | Local (macOS/Linux) |
| Nate | `04_causal_inference.py` | Local (macOS/Linux) |
| Tiffany | `05_figures.py` | Local (macOS/Linux) |

---

## Setup (Local)

**1. Create and activate the conda environment:**

```bash
conda env create -f environment.yml
conda activate causal-precip
```

**2. Install any remaining pip-only dependencies:**

```bash
pip install -r requirements.txt
```

**3. Configure the CDS API key** (required for `01_download_era5.py`):

Create `~/.cdsapirc` with your UID and API key from https://cds.climate.copernicus.eu:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API-KEY>
```

---

## Setup (Nautilus)

Kubernetes jobs in `k8s/` handle all environment setup automatically. Each job clones the repository at runtime, installs dependencies from `requirements.txt`, and runs the corresponding script with `ENV=nautilus`. Team members without Nautilus access do not need to interact with the `k8s/` directory. Before submitting a job, edit the `namespace` and `claimName` placeholders in the relevant YAML file to match your Nautilus project.

---

## Running the Pipeline

### Local

Run each script from the repo root. The `ENV` variable defaults to `local`, which uses the reduced date range and domain from `config/local.yaml`:

```bash
python scripts/01_download_era5.py
python scripts/02_preprocess.py
python scripts/03_causal_discovery.py
python scripts/04_causal_inference.py
python scripts/05_figures.py
```

### Nautilus

Submit each job individually after filling in the placeholder values:

```bash
kubectl apply -f k8s/01_download.yaml
kubectl apply -f k8s/02_preprocess.yaml
kubectl apply -f k8s/03_discovery.yaml
kubectl apply -f k8s/04_inference.yaml
kubectl apply -f k8s/05_figures.yaml
```

Monitor job status with `kubectl get jobs -n YOUR_NAMESPACE` and retrieve logs with `kubectl logs -n YOUR_NAMESPACE job/causal-precip-NN-jobname`.

---

## Configuration

Three YAML files control pipeline behaviour:

| File | Purpose |
|------|---------|
| `config/base.yaml` | Shared defaults inherited by all environments (full domain, full date range, all variables) |
| `config/local.yaml` | Developer override — smaller domain and date range so the pipeline finishes quickly on a laptop |
| `config/nautilus.yaml` | Kubernetes override — full domain with paths pointing to the PVC mount at `/workspace/` |

The active environment is selected by the `ENV` environment variable:

```bash
ENV=local python scripts/02_preprocess.py    # uses local.yaml (default)
ENV=nautilus python scripts/02_preprocess.py # uses nautilus.yaml
```

`load_config()` deep-merges the base config with the environment-specific override; nested keys not present in the override are preserved from the base.

---

## Contributing

Branch off `main` before starting work — each team member owns one branch named after their script (e.g., `01-download`, `03-discovery`). Commit early and often within your branch. Open a pull request into `main` when your script is ready for review; PRs require approval from at least one other team member before merging.
