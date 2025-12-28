# Gas Sensor Array Drift: KD vs DRCA (Reproducibility Package)

This repository provides code to reproduce the experiments in our manuscript:

**Sensor Drift Compensation in Electronic-Nose-Based Gas Recognition Using Knowledge Distillation**

It evaluates three drift-compensation settings on the UCI Gas Sensor Array Drift Dataset:
- **KD** (Knowledge Distillation)
- **DRCA** (Domain Regularized Component Analysis)
- **KD-DRCA** (hybrid)

We report results over repeated random splits and provide scripts/notebooks for:
- **Task 1**: train on Batch 1 → test on Batch 2–10
- **Task 2**: train on Batches 1–(n−1) → test on Batch n (n=2…10)

## Repository structure

- `notebooks/`  
  Jupyter notebooks used during development and for interactive reproduction.
- `src/`  
  Minimal Python scripts to run Task 1 / Task 2 and export results.
- `data/raw/`  
  **Place the dataset files here** (e.g., `batch1.dat` … `batch10.dat`).  
  The dataset is not included in this repository.
- `outputs/`  
  Generated plots and Excel files will be written here.

## 1) Download the dataset

Download the **Gas Sensor Array Drift Dataset** from the UCI Machine Learning Repository,
and place the files into:

```
data/raw/
  batch1.dat
  batch2.dat
  ...
  batch10.dat
```

## 2) Setup

Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## 3) Run experiments (scripts)

Task 1:
```bash
python src/run_task1.py
```

Task 2:
```bash
python src/run_task2.py
```

Each script will:
- perform standardization (global scaler fit on all batches)
- run repeated validation/test splits
- export detailed and summary results to Excel under `outputs/`
- optionally generate plots under `outputs/`

## 4) Run experiments (notebooks)

Open:
- `notebooks/task1_one_to_one.ipynb`
- `notebooks/task2_incremental.ipynb`

and run cells from top to bottom.

## Notes on reproducibility

- We recommend fixing random seeds for NumPy / TensorFlow and logging them.
- The dataset is **not** committed to this repo; it is referenced and loaded locally.
- If you generate large output files, consider excluding them from version control.

## Citation

If you use this code, please cite our paper (to be updated with DOI upon acceptance).

## License

MIT License (see `LICENSE`).
