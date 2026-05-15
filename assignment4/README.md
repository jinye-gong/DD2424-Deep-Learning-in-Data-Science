# DD2424 Assignment 4

Vanilla RNN character-level language model on *Harry Potter and the Goblet of Fire* (`goblet_book.txt`).

## Main files

- `Assignment4.py`: data prep, forward/backward (BPTT), Adam training, synthesis, gradient check, checkpoint save/load.
- `Assignment4_bonus.py`: Exercise 2 bonus (training order, batch-8, temperature/nucleus sampling, fast gradients).
- `torch_gradient_computations_column_wise.py`: PyTorch autograd reference (column storage; used for grad check).
- `torch_gradient_computations_row_wise.py`: row-storage variant.
- `report.md` / `report.pdf`: main assignment report.
- `report_bonus.md` / `report_bonus.pdf`: bonus report.
- `Assignment4.pdf`: course handout.

## Result folders

- `results/`: main run (100k updates) — `rnn_params.npz`, `char_vocab.txt`, `smooth_loss.png`, `synth_samples.txt`, `final_synthesis_1000.txt`.
- `results_bonus/`: bonus experiments — `bonus_results.json`, plots, `bonus_sampling_samples.txt`.

**Note:** Always load `unique_chars` from the checkpoint (`rnn_params.npz` or `char_vocab.txt`) when synthesizing; do not rebuild vocab from `list(set(book_data))` in a new process.

## Run commands

Use conda env `dd2424` (NumPy, PyTorch, matplotlib).

```bash
cd assignment4

# Gradient check only
conda run -n dd2424 python Assignment4.py --mode gradcheck

# Training only (default: 100,000 updates)
conda run -n dd2424 python Assignment4.py --mode train --max-updates 100000

# Grad check + train
conda run -n dd2424 python Assignment4.py --mode all --max-updates 100000

# Bonus (all Exercise 2 items)
conda run -n dd2424 python Assignment4_bonus.py --mode all --max-updates 15000
```

## Hyper-parameters (defaults)

| Parameter | Value |
|---|---:|
| hidden size m | 100 |
| learning rate eta | 0.001 |
| sequence length | 25 |
| optimizer | Adam |
| vocabulary K | 80 |

## Report PDF

Reports use plain text for math (no `$...$`) so PDF export does not drop variables or numbers. Regenerate PDFs with pandoc + a PDF engine, or open the `.md` files in your editor and export to PDF.
