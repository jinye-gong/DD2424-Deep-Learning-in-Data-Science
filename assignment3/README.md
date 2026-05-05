# DD2424 Assignment 3

Three-layer CIFAR-10 classifier with an initial patchify convolution layer.

## Main files

- `Assignment3.py`: full implementation for debug check, Exercise 3/4, and bonus experiments.
- `report.md` / `report.pdf`: main assignment report.
- `report_bonus.md` / `report_bonus.pdf`: bonus report.

## Result folders

- `results_assignment3/`: full non-quick results for Exercise 3/4.
- `results_assignment3_bonus/`: full bonus results (Exercise 5.1 and 5.2).

## Run commands

- Main experiments:
  - `conda run -n dd2424 python Assignment3.py --mode all --cifar-root Datasets/cifar-10-batches-py --out-dir results_assignment3`
- Bonus experiments:
  - `conda run -n dd2424 python Assignment3.py --mode bonus --cifar-root Datasets/cifar-10-batches-py --out-dir results_assignment3_bonus`
