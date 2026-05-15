from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Assignment4 import (
    adam_update,
    build_char_mappings,
    chars_to_onehot,
    init_adam_state,
    init_rnn,
    load_rnn_checkpoint,
    read_book_data,
    save_char_vocab,
    save_rnn_checkpoint,
    softmax,
)

TrainingMode = Literal["sequential", "random_chunks", "random_position"]


def chars_to_indices(chars: str, char_to_ind: Dict[str, int]) -> np.ndarray:
    return np.array([char_to_ind[ch] for ch in chars], dtype=np.int64)


def split_book_chunks(
    book_data: str, num_chunks: int, val_chunk_idx: int = 0
) -> Tuple[str, List[str]]:
    n = len(book_data)
    chunk_size = n // num_chunks
    chunks: List[str] = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else n
        chunks.append(book_data[start:end])
    val_data = chunks[val_chunk_idx]
    train_chunks = [c for j, c in enumerate(chunks) if j != val_chunk_idx]
    return val_data, train_chunks


def softmax_temperature(o: np.ndarray, temperature: float) -> np.ndarray:
    o_shift = o - np.max(o, axis=0, keepdims=True)
    exp_o = np.exp(o_shift / temperature)
    return exp_o / np.sum(exp_o, axis=0, keepdims=True)


def nucleus_distribution(p: np.ndarray, theta: float) -> np.ndarray:
    flat = p.ravel()
    order = np.argsort(flat)[::-1]
    sorted_p = flat[order]
    cum = np.cumsum(sorted_p)
    k = int(np.searchsorted(cum, theta) + 1)
    keep = order[:k]
    p_tilde = np.zeros_like(flat)
    p_tilde[keep] = flat[keep]
    p_tilde /= np.sum(p_tilde)
    return p_tilde.reshape(p.shape)


def sample_from_p(p: np.ndarray, rng: np.random.Generator) -> int:
    cp = np.cumsum(p, axis=0)
    a = rng.uniform(size=1)
    return int(np.argmax(cp - a > 0))


def forward_pass_fast(
    x_inds: np.ndarray,
    y_inds: np.ndarray,
    h0: np.ndarray,
    RNN: Dict[str, np.ndarray],
) -> Tuple[float, Dict[str, List[np.ndarray]], np.ndarray]:
    tau = len(x_inds)
    h_list: List[np.ndarray] = []
    h_prev_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []

    hprev = h0.copy()
    loss = 0.0
    for t in range(tau):
        h_prev_list.append(hprev.copy())
        a = RNN["W"] @ hprev + RNN["U"][:, [x_inds[t]]] + RNN["b"]
        h = np.tanh(a)
        o = RNN["V"] @ h + RNN["c"]
        p = softmax(o)
        loss += -np.log(p[y_inds[t], 0] + 1e-12)
        h_list.append(h)
        p_list.append(p)
        hprev = h

    loss /= tau
    fp_data = {"h": h_list, "h_prev": h_prev_list, "p": p_list}
    return loss, fp_data, hprev


def backward_pass_fast(
    x_inds: np.ndarray,
    y_inds: np.ndarray,
    fp_data: Dict[str, List[np.ndarray]],
    RNN: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    tau = len(x_inds)
    grads = {kk: np.zeros_like(RNN[kk]) for kk in RNN}
    delta_a_next = np.zeros((RNN["b"].shape[0], 1), dtype=np.float64)

    for t in reversed(range(tau)):
        p = fp_data["p"][t]
        delta_o = p.copy()
        delta_o /= tau
        delta_o[y_inds[t], 0] -= 1.0 / tau

        h_t = fp_data["h"][t]
        h_prev = fp_data["h_prev"][t]
        delta_h = RNN["V"].T @ delta_o + delta_a_next
        delta_a = delta_h * (1.0 - h_t**2)

        grads["V"] += delta_o @ h_t.T
        grads["c"] += delta_o
        grads["W"] += np.outer(delta_a.ravel(), h_prev.ravel()).reshape(RNN["W"].shape)
        grads["U"][:, [x_inds[t]]] += delta_a
        grads["b"] += delta_a
        delta_a_next = RNN["W"].T @ delta_a

    return grads


def forward_pass_slow(
    X: np.ndarray,
    Y: np.ndarray,
    h0: np.ndarray,
    RNN: Dict[str, np.ndarray],
) -> Tuple[float, Dict[str, List[np.ndarray]], np.ndarray]:
    tau = X.shape[1]
    h_list: List[np.ndarray] = []
    h_prev_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []
    hprev = h0.copy()
    loss = 0.0
    for t in range(tau):
        h_prev_list.append(hprev.copy())
        a = RNN["W"] @ hprev + RNN["U"] @ X[:, t : t + 1] + RNN["b"]
        h = np.tanh(a)
        o = RNN["V"] @ h + RNN["c"]
        p = softmax(o)
        loss += float(-np.sum(Y[:, t : t + 1] * np.log(p + 1e-12)))
        h_list.append(h)
        p_list.append(p)
        hprev = h
    loss /= tau
    return loss, {"h": h_list, "h_prev": h_prev_list, "p": p_list}, hprev


def backward_pass_slow(
    X: np.ndarray,
    Y: np.ndarray,
    fp_data: Dict[str, List[np.ndarray]],
    RNN: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    tau = X.shape[1]
    grads = {kk: np.zeros_like(RNN[kk]) for kk in RNN}
    delta_a_next = np.zeros((RNN["b"].shape[0], 1), dtype=np.float64)
    for t in reversed(range(tau)):
        p = fp_data["p"][t]
        delta_o = (p - Y[:, t : t + 1]) / tau
        h_t = fp_data["h"][t]
        h_prev = fp_data["h_prev"][t]
        delta_h = RNN["V"].T @ delta_o + delta_a_next
        delta_a = delta_h * (1.0 - h_t**2)
        grads["V"] += delta_o @ h_t.T
        grads["c"] += delta_o
        grads["W"] += delta_a @ h_prev.T
        grads["U"] += delta_a @ X[:, t : t + 1].T
        grads["b"] += delta_a
        delta_a_next = RNN["W"].T @ delta_a
    return grads


def eval_validation_loss(
    val_data: str,
    char_to_ind: Dict[str, int],
    RNN: Dict[str, np.ndarray],
    seq_length: int,
    m: int,
) -> float:
    max_start = len(val_data) - seq_length - 1
    if max_start <= 0:
        return float("nan")
    losses = []
    h0 = np.zeros((m, 1), dtype=np.float64)
    for e in range(0, max_start, seq_length):
        x_inds = chars_to_indices(val_data[e : e + seq_length], char_to_ind)
        y_inds = chars_to_indices(val_data[e + 1 : e + seq_length + 1], char_to_ind)
        loss, _, _ = forward_pass_fast(x_inds, y_inds, h0, RNN)
        losses.append(loss)
    return float(np.mean(losses))


def train_one_mode(
    train_text: str,
    char_to_ind: Dict[str, int],
    val_data: str,
    rng: np.random.Generator,
    mode: TrainingMode,
    max_updates: int,
    m: int,
    eta: float,
    seq_length: int,
    train_chunks: Optional[List[str]] = None,
    batch_size: int = 1,
    init_rnn_state: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], List[int], List[float], List[float]]:
    if init_rnn_state is not None:
        RNN = {kk: init_rnn_state[kk].copy() for kk in init_rnn_state}
    else:
        RNN = init_rnn(len(char_to_ind), m, rng)
    m_states, v_states = init_adam_state(RNN)

    val_every = max(500, max_updates // 25)
    val_steps: List[int] = []
    val_losses: List[float] = []
    train_smooth: List[float] = []
    train_steps: List[int] = []

    smooth_loss = -np.log(1.0 / len(char_to_ind))
    e = 0
    hprev = np.zeros((m, 1), dtype=np.float64)
    chunk_queue: List[int] = []
    current_chunk_idx = -1
    current_chunk = train_text

    for t0 in range(1, max_updates + 1):
        if mode == "sequential":
            if e == 0:
                hprev = np.zeros((m, 1), dtype=np.float64)
            x_inds = chars_to_indices(train_text[e : e + seq_length], char_to_ind)
            y_inds = chars_to_indices(train_text[e + 1 : e + seq_length + 1], char_to_ind)
            e += seq_length
            if e > len(train_text) - seq_length - 1:
                e = 0

        elif mode == "random_chunks":
            assert train_chunks is not None
            if current_chunk_idx == -1 or e > len(current_chunk) - seq_length - 1:
                if not chunk_queue:
                    chunk_queue = rng.permutation(len(train_chunks)).tolist()
                current_chunk_idx = chunk_queue.pop(0)
                current_chunk = train_chunks[current_chunk_idx]
                e = 0
                hprev = np.zeros((m, 1), dtype=np.float64)
            x_inds = chars_to_indices(current_chunk[e : e + seq_length], char_to_ind)
            y_inds = chars_to_indices(current_chunk[e + 1 : e + seq_length + 1], char_to_ind)
            e += seq_length

        else:  # random_position
            max_start = len(train_text) - seq_length - 1
            e_rand = int(rng.integers(0, max_start + 1))
            x_inds = chars_to_indices(train_text[e_rand : e_rand + seq_length], char_to_ind)
            y_inds = chars_to_indices(train_text[e_rand + 1 : e_rand + seq_length + 1], char_to_ind)
            hprev = np.zeros((m, 1), dtype=np.float64)

        loss, fp, hprev = forward_pass_fast(x_inds, y_inds, hprev, RNN)
        grads = backward_pass_fast(x_inds, y_inds, fp, RNN)

        if batch_size > 1 and mode == "random_position":
            total_loss = loss
            for _ in range(batch_size - 1):
                e_b = int(rng.integers(0, len(train_text) - seq_length - 1))
                x_b = chars_to_indices(train_text[e_b : e_b + seq_length], char_to_ind)
                y_b = chars_to_indices(train_text[e_b + 1 : e_b + seq_length + 1], char_to_ind)
                h0 = np.zeros((m, 1), dtype=np.float64)
                loss_b, fp_b, _ = forward_pass_fast(x_b, y_b, h0, RNN)
                g_b = backward_pass_fast(x_b, y_b, fp_b, RNN)
                total_loss += loss_b
                for kk in grads:
                    grads[kk] += g_b[kk]
            loss = total_loss / batch_size
            for kk in grads:
                grads[kk] /= batch_size

        adam_update(RNN, grads, m_states, v_states, t0, eta)
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        train_smooth.append(smooth_loss)
        train_steps.append(t0)

        if t0 % val_every == 0 or t0 == max_updates:
            val_steps.append(t0)
            val_losses.append(eval_validation_loss(val_data, char_to_ind, RNN, seq_length, m))

    return RNN, train_steps, train_smooth, val_steps, val_losses


def benchmark_gradients(
    book_data: str,
    char_to_ind: Dict[str, int],
    K: int,
    rng: np.random.Generator,
    m: int = 100,
    seq_length: int = 25,
    repeats: int = 400,
) -> Dict[str, float]:
    X_chars = book_data[0:seq_length]
    Y_chars = book_data[1 : seq_length + 1]
    X = chars_to_onehot(X_chars, char_to_ind, K)
    Y = chars_to_onehot(Y_chars, char_to_ind, K)
    x_inds = chars_to_indices(X_chars, char_to_ind)
    y_inds = chars_to_indices(Y_chars, char_to_ind)
    h0 = np.zeros((m, 1), dtype=np.float64)
    RNN = init_rnn(K, m, rng)

    for _ in range(5):
        _, fp_s, _ = forward_pass_slow(X, Y, h0, RNN)
        backward_pass_slow(X, Y, fp_s, RNN)
        _, fp_f, _ = forward_pass_fast(x_inds, y_inds, h0, RNN)
        backward_pass_fast(x_inds, y_inds, fp_f, RNN)

    t0 = time.perf_counter()
    for _ in range(repeats):
        _, fp_s, _ = forward_pass_slow(X, Y, h0, RNN)
        backward_pass_slow(X, Y, fp_s, RNN)
    slow_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(repeats):
        _, fp_f, _ = forward_pass_fast(x_inds, y_inds, h0, RNN)
        backward_pass_fast(x_inds, y_inds, fp_f, RNN)
    fast_s = time.perf_counter() - t0

    torch_s = float("nan")
    try:
        from torch_gradient_computations_column_wise import ComputeGradsWithTorch

        for _ in range(5):
            ComputeGradsWithTorch(X, y_inds, h0, RNN)
        t0 = time.perf_counter()
        for _ in range(repeats):
            ComputeGradsWithTorch(X, y_inds, h0, RNN)
        torch_s = time.perf_counter() - t0
    except Exception:
        pass

    return {
        "repeats": repeats,
        "slow_onehot_s": slow_s,
        "fast_indexed_s": fast_s,
        "torch_s": torch_s,
        "speedup_fast_vs_slow": slow_s / fast_s,
        "ratio_slow_vs_torch": slow_s / torch_s if np.isfinite(torch_s) else None,
        "ratio_fast_vs_torch": fast_s / torch_s if np.isfinite(torch_s) else None,
    }


def synthesize_text(
    RNN: Dict[str, np.ndarray],
    ind_to_char: Dict[int, str],
    h0: np.ndarray,
    x0: np.ndarray,
    n: int,
    rng: np.random.Generator,
    temperature: Optional[float] = None,
    nucleus_theta: Optional[float] = None,
) -> str:
    hprev = h0.copy()
    x = x0.copy()
    out: List[str] = []
    for _ in range(n):
        a = RNN["W"] @ hprev + RNN["U"] @ x + RNN["b"]
        h = np.tanh(a)
        o = RNN["V"] @ h + RNN["c"]
        if temperature is not None:
            p = softmax_temperature(o, temperature)
        else:
            p = softmax(o)
        if nucleus_theta is not None:
            p = nucleus_distribution(p, nucleus_theta)
        idx = sample_from_p(p, rng)
        out.append(ind_to_char[idx])
        x = np.zeros_like(x)
        x[idx, 0] = 1.0
        hprev = h
    return "".join(out)


def run_training_comparison(
    book_data: str,
    char_to_ind: Dict[str, int],
    val_data: str,
    train_chunks: List[str],
    train_text: str,
    rng: np.random.Generator,
    out_dir: str,
    max_updates: int,
    m: int,
    eta: float,
    seq_length: int,
) -> Dict:
    results: Dict = {"max_updates": max_updates}
    modes: List[Tuple[str, TrainingMode, int]] = [
        ("sequential", "sequential", 1),
        ("random_chunks", "random_chunks", 1),
        ("random_position", "random_position", 1),
        ("random_position_batch8", "random_position", 8),
    ]

    K = len(char_to_ind)
    init_rng = np.random.default_rng(100)
    rnn_template = init_rnn(K, m, init_rng)
    strategy_seeds = {
        "sequential": 201,
        "random_chunks": 202,
        "random_position": 203,
        "random_position_batch8": 204,
    }

    plt.figure(figsize=(9, 5))
    for label, mode, batch_size in modes:
        print(f"[train] {label} ...")
        rnn_init = {kk: rnn_template[kk].copy() for kk in rnn_template}
        train_rng = np.random.default_rng(strategy_seeds[label])
        _, train_steps, train_smooth, val_steps, val_losses = train_one_mode(
            train_text,
            char_to_ind,
            val_data,
            train_rng,
            mode,
            max_updates,
            m,
            eta,
            seq_length,
            train_chunks=train_chunks if mode == "random_chunks" else None,
            batch_size=batch_size,
            init_rnn_state=rnn_init,
        )
        results[label] = {
            "final_train_smooth": float(train_smooth[-1]),
            "final_val_loss": float(val_losses[-1]),
            "val_steps": val_steps,
            "val_losses": val_losses,
        }
        plt.plot(val_steps, val_losses, label=label)

    plt.xlabel("update step")
    plt.ylabel("validation loss (mean CE)")
    plt.title("Training strategy comparison (held-out chunk)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bonus_val_loss_compare.png"), dpi=150)
    plt.close()
    return results


def run_sampling_demo(
    RNN: Dict[str, np.ndarray],
    ind_to_char: Dict[int, str],
    char_to_ind: Dict[str, int],
    K: int,
    rng: np.random.Generator,
    out_dir: str,
    n_chars: int = 250,
) -> Dict:
    h0 = np.zeros((RNN["b"].shape[0], 1), dtype=np.float64)
    x0 = np.zeros((K, 1), dtype=np.float64)
    x0[char_to_ind["."], 0] = 1.0

    samples: Dict[str, str] = {}
    temperatures = {"low_T0.5": 0.5, "medium_T1.0": 1.0, "high_T1.5": 1.5}
    thetas = {"low_theta0.5": 0.5, "medium_theta0.9": 0.9, "high_theta0.99": 0.99}

    for name, T in temperatures.items():
        text = synthesize_text(RNN, ind_to_char, h0, x0, n_chars, rng, temperature=T)
        samples[name] = text
        print(f"[sampling] {name}: {text[:120]}...")

    for name, theta in thetas.items():
        text = synthesize_text(RNN, ind_to_char, h0, x0, n_chars, rng, nucleus_theta=theta)
        samples[name] = text
        print(f"[sampling] {name}: {text[:120]}...")

    with open(os.path.join(out_dir, "bonus_sampling_samples.txt"), "w", encoding="utf-8") as f:
        for k, v in samples.items():
            f.write(f"=== {k} ===\n{v}\n\n")

    return {"temperature": temperatures, "nucleus_theta": thetas, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="DD2424 Assignment 4 Bonus")
    parser.add_argument(
        "--book",
        default=os.path.join(os.path.dirname(__file__), "goblet_book.txt"),
    )
    parser.add_argument(
        "--params",
        default=os.path.join(os.path.dirname(__file__), "results", "rnn_params.npz"),
        help="Trained RNN from base assignment (for sampling demo)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "results_bonus"),
    )
    parser.add_argument(
        "--mode",
        choices=["all", "train_compare", "benchmark", "sampling"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-updates", type=int, default=15_000)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--seq-length", type=int, default=25)
    parser.add_argument("--num-chunks", type=int, default=20)
    parser.add_argument("--benchmark-repeats", type=int, default=400)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    book_data = read_book_data(args.book)
    _, char_to_ind, ind_to_char, K = build_char_mappings(book_data)
    val_data, train_chunks = split_book_chunks(book_data, args.num_chunks, val_chunk_idx=0)
    train_text = "".join(train_chunks)
    print(f"book={len(book_data)} chars, K={K}, val_chunk={len(val_data)} chars")

    all_results: Dict = {}

    if args.mode in ("all", "benchmark"):
        print("[benchmark] gradient speed ...")
        bench = benchmark_gradients(
            book_data, char_to_ind, K, rng, m=args.m, seq_length=args.seq_length, repeats=args.benchmark_repeats
        )
        all_results["benchmark"] = bench
        print(json.dumps(bench, indent=2))

        labels = ["slow one-hot", "fast indexed", "PyTorch"]
        times = [bench["slow_onehot_s"], bench["fast_indexed_s"], bench["torch_s"]]
        valid = [(l, t) for l, t in zip(labels, times) if np.isfinite(t)]
        plt.figure(figsize=(7, 4))
        plt.bar([x[0] for x in valid], [x[1] for x in valid])
        plt.ylabel(f"time for {bench['repeats']} forward+backward passes (s)")
        plt.title("Gradient computation speed")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "bonus_benchmark_timing.png"), dpi=150)
        plt.close()

    if args.mode in ("all", "train_compare"):
        print("[train_compare] strategies ...")
        train_res = run_training_comparison(
            book_data,
            char_to_ind,
            val_data,
            train_chunks,
            train_text,
            rng,
            args.out_dir,
            args.max_updates,
            args.m,
            args.eta,
            args.seq_length,
        )
        all_results["training_comparison"] = train_res

    if args.mode in ("all", "sampling"):
        print("[sampling] temperature & nucleus ...")
        vocab_path = os.path.join(os.path.dirname(args.params), "char_vocab.txt")
        if os.path.isfile(args.params):
            RNN, char_to_ind, ind_to_char, K = load_rnn_checkpoint(
                args.params, book_data=book_data, vocab_path=vocab_path
            )
            print(f"[sampling] loaded checkpoint with K={K} (vocab from checkpoint/vocab file)")
        else:
            print(f"[sampling] params not found at {args.params}, training 5k steps ...")
            RNN, _, _, _, _ = train_one_mode(
                train_text,
                char_to_ind,
                val_data,
                rng,
                "random_position",
                5000,
                args.m,
                args.eta,
                args.seq_length,
                batch_size=8,
            )
            _, char_to_ind, ind_to_char, K = build_char_mappings(book_data)
        sampling_res = run_sampling_demo(RNN, ind_to_char, char_to_ind, K, rng, args.out_dir)
        all_results["sampling"] = {k: sampling_res[k] for k in ("temperature", "nucleus_theta")}

    out_json = os.path.join(args.out_dir, "bonus_results.json")
    if os.path.isfile(out_json):
        with open(out_json, encoding="utf-8") as f:
            prev = json.load(f)
        prev.update(all_results)
        all_results = prev

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        raise TypeError(type(obj))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
