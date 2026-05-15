from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_book_data(book_fname: str) -> str:
    with open(book_fname, "r", encoding="utf-8") as fid:
        return fid.read()


def build_char_mappings(book_data: str) -> Tuple[List[str], Dict[str, int], Dict[int, str], int]:
    unique_chars = list(set(book_data))
    K = len(unique_chars)
    char_to_ind = {ch: unique_chars.index(ch) for ch in unique_chars}
    ind_to_char = {i: unique_chars[i] for i in range(K)}
    return unique_chars, char_to_ind, ind_to_char, K


def save_char_vocab(path: str, unique_chars: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ch in unique_chars:
            if ch == "\n":
                f.write("\n")  # blank line = newline character
            else:
                f.write(ch + "\n")


def load_char_vocab(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return []
    # One character per line; blank line encodes newline.
    out: List[str] = []
    for line in lines:
        if line == "\n":
            out.append("\n")
        else:
            out.append(line.rstrip("\n"))
    return out


def mappings_from_unique_chars(
    unique_chars: List[str],
) -> Tuple[Dict[str, int], Dict[int, str], int]:
    char_to_ind = {ch: i for i, ch in enumerate(unique_chars)}
    ind_to_char = {i: ch for i, ch in enumerate(unique_chars)}
    return char_to_ind, ind_to_char, len(unique_chars)


def save_rnn_checkpoint(
    path: str,
    RNN: Dict[str, np.ndarray],
    unique_chars: List[str],
    smooth_loss: np.ndarray,
    update_steps: np.ndarray,
) -> None:
    np.savez(
        path,
        **{kk: RNN[kk] for kk in RNN},
        unique_chars=np.array(unique_chars, dtype=object),
        smooth_loss=smooth_loss,
        update_steps=update_steps,
    )


def load_rnn_checkpoint(
    path: str,
    book_data: Optional[str] = None,
    vocab_path: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[int, str], int]:
    data = np.load(path, allow_pickle=True)
    RNN = {kk: data[kk] for kk in ("b", "c", "U", "W", "V")}

    unique_chars: Optional[List[str]] = None
    if "unique_chars" in data.files:
        unique_chars = [str(ch) for ch in data["unique_chars"].tolist()]

    if unique_chars is None and vocab_path and os.path.isfile(vocab_path):
        unique_chars = load_char_vocab(vocab_path)

    if unique_chars is None:
        if book_data is None:
            raise ValueError(
                "Checkpoint has no 'unique_chars'. Provide --book or a char_vocab.txt "
                "saved alongside the checkpoint."
            )
        print(
            "[warn] Checkpoint missing 'unique_chars'; rebuilding from book "
            "with list(set(book_data)). Re-save checkpoint after retraining if needed."
        )
        unique_chars, _, _, _ = build_char_mappings(book_data)

    K = len(unique_chars)
    if RNN["c"].shape[0] != K:
        raise ValueError(
            f"Vocab size {K} does not match RNN['c'] rows {RNN['c'].shape[0]}."
        )
    char_to_ind, ind_to_char, _ = mappings_from_unique_chars(unique_chars)
    return RNN, char_to_ind, ind_to_char, K


def chars_to_onehot(chars: str, char_to_ind: Dict[str, int], K: int) -> np.ndarray:
    tau = len(chars)
    X = np.zeros((K, tau), dtype=np.float64)
    for t, ch in enumerate(chars):
        X[char_to_ind[ch], t] = 1.0
    return X


def onehot_to_chars(Y: np.ndarray, ind_to_char: Dict[int, str]) -> str:
    indices = np.argmax(Y, axis=0)
    return "".join(ind_to_char[int(i)] for i in indices)


def init_rnn(K: int, m: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    RNN: Dict[str, np.ndarray] = {}
    RNN["b"] = np.zeros((m, 1), dtype=np.float64)
    RNN["c"] = np.zeros((K, 1), dtype=np.float64)
    RNN["U"] = (1.0 / np.sqrt(2 * K)) * rng.standard_normal((m, K))
    RNN["W"] = (1.0 / np.sqrt(2 * m)) * rng.standard_normal((m, m))
    RNN["V"] = (1.0 / np.sqrt(m)) * rng.standard_normal((K, m))
    return RNN


def softmax(o: np.ndarray) -> np.ndarray:
    o_shift = o - np.max(o, axis=0, keepdims=True)
    exp_o = np.exp(o_shift)
    return exp_o / np.sum(exp_o, axis=0, keepdims=True)


def sample_char(p: np.ndarray, rng: np.random.Generator) -> int:
    cp = np.cumsum(p, axis=0)
    a = rng.uniform(size=1)
    ii = int(np.argmax(cp - a > 0))
    return ii


def synthesize(
    RNN: Dict[str, np.ndarray],
    h0: np.ndarray,
    x0: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    K = RNN["c"].shape[0]
    Y = np.zeros((K, n), dtype=np.float64)
    hprev = h0.copy()
    x = x0.copy()

    for t in range(n):
        a = RNN["W"] @ hprev + RNN["U"] @ x + RNN["b"]
        h = np.tanh(a)
        o = RNN["V"] @ h + RNN["c"]
        p = softmax(o)
        ii = sample_char(p, rng)
        Y[ii, t] = 1.0
        x = Y[:, t : t + 1]
        hprev = h

    return Y


def forward_pass(
    X: np.ndarray,
    Y: np.ndarray,
    h0: np.ndarray,
    RNN: Dict[str, np.ndarray],
) -> Tuple[float, Dict[str, List[np.ndarray]], np.ndarray]:
    tau = X.shape[1]
    a_list: List[np.ndarray] = []
    h_list: List[np.ndarray] = []
    h_prev_list: List[np.ndarray] = []
    o_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []

    hprev = h0.copy()
    loss = 0.0
    for t in range(tau):
        h_prev_list.append(hprev.copy())
        a = RNN["W"] @ hprev + RNN["U"] @ X[:, t : t + 1] + RNN["b"]
        h = np.tanh(a)
        o = RNN["V"] @ h + RNN["c"]
        p = softmax(o)
        y_t = Y[:, t : t + 1]
        loss += float(-np.sum(y_t * np.log(p + 1e-12)))
        a_list.append(a)
        h_list.append(h)
        o_list.append(o)
        p_list.append(p)
        hprev = h

    loss /= tau
    fp_data = {"a": a_list, "h": h_list, "h_prev": h_prev_list, "o": o_list, "p": p_list}
    return loss, fp_data, hprev


def backward_pass(
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
        y_t = Y[:, t : t + 1]
        delta_o = (p - y_t) / tau

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


def init_adam_state(RNN: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    m_states = {kk: np.zeros_like(RNN[kk]) for kk in RNN}
    v_states = {kk: np.zeros_like(RNN[kk]) for kk in RNN}
    return m_states, v_states


def adam_update(
    RNN: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    m_states: Dict[str, np.ndarray],
    v_states: Dict[str, np.ndarray],
    t0: int,
    eta: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    for kk in RNN:
        m_states[kk] = beta1 * m_states[kk] + (1.0 - beta1) * grads[kk]
        v_states[kk] = beta2 * v_states[kk] + (1.0 - beta2) * (grads[kk] ** 2)
        m_hat = m_states[kk] / (1.0 - beta1**t0)
        v_hat = v_states[kk] / (1.0 - beta2**t0)
        RNN[kk] -= eta * m_hat / (np.sqrt(v_hat) + eps)


def gradient_check(
    book_data: str,
    char_to_ind: Dict[str, int],
    K: int,
    rng: np.random.Generator,
    seq_length: int = 25,
    m: int = 10,
) -> None:
    from torch_gradient_computations_column_wise import ComputeGradsWithTorch

    X_chars = book_data[0:seq_length]
    Y_chars = book_data[1 : seq_length + 1]
    X = chars_to_onehot(X_chars, char_to_ind, K)
    Y = chars_to_onehot(Y_chars, char_to_ind, K)
    y = np.array([char_to_ind[ch] for ch in Y_chars], dtype=np.int64)

    h0 = np.zeros((m, 1), dtype=np.float64)
    RNN = init_rnn(K, m, rng)

    loss, fp_data, _ = forward_pass(X, Y, h0, RNN)
    grads_analytic = backward_pass(X, Y, fp_data, RNN)
    grads_torch = ComputeGradsWithTorch(X, y, h0, RNN)

    eps = 1e-12
    print(f"[gradient_check] forward loss={loss:.6f}")
    for kk in RNN:
        rel = np.max(
            np.abs(grads_analytic[kk] - grads_torch[kk])
            / np.maximum(eps, np.abs(grads_analytic[kk]) + np.abs(grads_torch[kk]))
        )
        print(f"[gradient_check] {kk}: max relative error = {rel:.3e}")


def train_rnn(
    book_data: str,
    unique_chars: List[str],
    char_to_ind: Dict[str, int],
    ind_to_char: Dict[int, str],
    K: int,
    rng: np.random.Generator,
    out_dir: str,
    m: int = 100,
    eta: float = 0.001,
    seq_length: int = 25,
    max_updates: int = 100_000,
    print_every: int = 100,
    synth_every: int = 1000,
    synth_len: int = 200,
    sample_every: int = 10_000,
) -> Dict[str, np.ndarray]:
    os.makedirs(out_dir, exist_ok=True)

    RNN = init_rnn(K, m, rng)
    m_states, v_states = init_adam_state(RNN)

    e = 0
    hprev = np.zeros((m, 1), dtype=np.float64)
    smooth_loss = -np.log(1.0 / K)
    t0 = 0

    loss_history: List[float] = []
    update_steps: List[int] = []
    synth_log: List[str] = []

    x0_char = "."
    x0 = np.zeros((K, 1), dtype=np.float64)
    x0[char_to_ind[x0_char], 0] = 1.0

    h0_zero = np.zeros((m, 1), dtype=np.float64)
    Y_syn = synthesize(RNN, h0_zero, x0, synth_len, rng)
    synth_log.append(
        f"iter = 0 (before training), smooth_loss={smooth_loss}\n"
        f"{onehot_to_chars(Y_syn, ind_to_char)}\n"
    )
    print(f"[synth @ 0 (before training)] {onehot_to_chars(Y_syn, ind_to_char)[:120]}...")

    while t0 < max_updates:
        t0 += 1
        X_chars = book_data[e : e + seq_length]
        Y_chars = book_data[e + 1 : e + seq_length + 1]
        X = chars_to_onehot(X_chars, char_to_ind, K)
        Y = chars_to_onehot(Y_chars, char_to_ind, K)

        if e == 0:
            hprev = np.zeros((m, 1), dtype=np.float64)
        h_for_synth = hprev.copy()

        if t0 % sample_every == 0:
            Y_syn = synthesize(RNN, h_for_synth, X[:, 0:1], synth_len, rng)
            synth_log.append(
                f"iter = {t0} (before update), smooth_loss={smooth_loss}\n"
                f"{onehot_to_chars(Y_syn, ind_to_char)}\n"
            )

        loss, fp_data, hprev = forward_pass(X, Y, hprev, RNN)
        grads = backward_pass(X, Y, fp_data, RNN)
        adam_update(RNN, grads, m_states, v_states, t0, eta)

        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        loss_history.append(smooth_loss)
        update_steps.append(t0)

        if t0 == 1 or t0 % print_every == 0:
            print(f"iter = {t0}, smooth_loss={smooth_loss}")

        if t0 % synth_every == 0:
            Y_syn = synthesize(RNN, hprev, X[:, 0:1], synth_len, rng)
            text = onehot_to_chars(Y_syn, ind_to_char)
            print(f"[synth @ {t0}] {text}")

        e += seq_length
        if e > len(book_data) - seq_length - 1:
            e = 0
            hprev = np.zeros((m, 1), dtype=np.float64)
            print(f"[epoch complete] after update {t0}")

    plt.figure(figsize=(8, 4))
    plt.plot(update_steps, loss_history, linewidth=1.0)
    plt.xlabel("update step")
    plt.ylabel("smooth loss")
    plt.title("RNN training smooth loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "smooth_loss.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "synth_samples.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(synth_log))

    Y_final = synthesize(RNN, np.zeros((m, 1)), x0, 1000, rng)
    final_text = onehot_to_chars(Y_final, ind_to_char)
    with open(os.path.join(out_dir, "final_synthesis_1000.txt"), "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"[final 1000-char sample]\n{final_text[:500]}...")

    ckpt_path = os.path.join(out_dir, "rnn_params.npz")
    vocab_path = os.path.join(out_dir, "char_vocab.txt")
    save_char_vocab(vocab_path, unique_chars)
    save_rnn_checkpoint(
        ckpt_path,
        RNN,
        unique_chars,
        np.array(loss_history),
        np.array(update_steps),
    )
    return RNN


def main() -> None:
    parser = argparse.ArgumentParser(description="DD2424 Assignment 4 - Vanilla RNN")
    parser.add_argument(
        "--book",
        default=os.path.join(os.path.dirname(__file__), "goblet_book.txt"),
        help="Path to goblet_book.txt",
    )
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--mode", choices=["gradcheck", "train", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-updates", type=int, default=100_000)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--seq-length", type=int, default=25)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    book_data = read_book_data(args.book)
    unique_chars, char_to_ind, ind_to_char, K = build_char_mappings(book_data)
    print(f"Loaded book: {len(book_data)} characters, K={K} unique chars")

    if args.mode in ("gradcheck", "all"):
        gradient_check(book_data, char_to_ind, K, rng, seq_length=args.seq_length, m=10)

    if args.mode in ("train", "all"):
        train_rnn(
            book_data,
            unique_chars,
            char_to_ind,
            ind_to_char,
            K,
            rng,
            args.out_dir,
            m=args.m,
            eta=args.eta,
            seq_length=args.seq_length,
            max_updates=args.max_updates,
        )


if __name__ == "__main__":
    main()
