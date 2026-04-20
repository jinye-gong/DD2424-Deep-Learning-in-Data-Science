import copy
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_batch(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one CIFAR-10 batch and return (X, Y, y).
    X: (d, n) float64 in [0, 1]
    Y: (K, n) one-hot float64
    y: (n,) int64 labels in [0, 9]
    """
    with open(filename, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")

    X = data_dict[b"data"].astype(np.float64) / 255.0  # (n, d)
    X = X.T  # (d, n)

    y = np.array(data_dict[b"labels"], dtype=np.int64)  # (n,)
    K = 10
    n = y.shape[0]
    Y = np.zeros((K, n), dtype=np.float64)
    Y[y, np.arange(n)] = 1.0
    return X, Y, y


def normalize_data(
    train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize using train mean/std per feature dimension.
    Returns normalized train/val/test and (mean_X, std_X).
    """
    mean_X = np.mean(train_X, axis=1, keepdims=True)
    std_X = np.std(train_X, axis=1, keepdims=True)
    std_X = np.where(std_X < 1e-12, 1.0, std_X)

    train_X_n = (train_X - mean_X) / std_X
    val_X_n = (val_X - mean_X) / std_X
    test_X_n = (test_X - mean_X) / std_X
    return train_X_n, val_X_n, test_X_n, mean_X, std_X


def load_all_data(cifar_root: str) -> Dict[str, np.ndarray]:
    """
    Assignment setup:
    - data_batch_1: training
    - data_batch_2: validation
    - test_batch: testing
    """
    train_X, train_Y, train_y = load_batch(os.path.join(cifar_root, "data_batch_1"))
    val_X, val_Y, val_y = load_batch(os.path.join(cifar_root, "data_batch_2"))
    test_X, test_Y, test_y = load_batch(os.path.join(cifar_root, "test_batch"))

    train_X, val_X, test_X, mean_X, std_X = normalize_data(train_X, val_X, test_X)
    return {
        "train_X": train_X,
        "train_Y": train_Y,
        "train_y": train_y,
        "val_X": val_X,
        "val_Y": val_Y,
        "val_y": val_y,
        "test_X": test_X,
        "test_Y": test_Y,
        "test_y": test_y,
        "mean_X": mean_X,
        "std_X": std_X,
    }


def figure_tag_config(lam: float, eta: float) -> str:
    """Filename tag using Greek letters (matches lam/eta in the assignment)."""
    return f"λ{lam}_η{eta}".replace(".", "p")


def init_network(K: int, d: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    return {
        "W": 0.01 * rng.standard_normal((K, d)),
        "b": np.zeros((K, 1), dtype=np.float64),
    }


def apply_network(X: np.ndarray, network: Dict[str, np.ndarray]) -> np.ndarray:
    s = network["W"] @ X + network["b"]
    s_shifted = s - np.max(s, axis=0, keepdims=True)
    exp_s = np.exp(s_shifted)
    P = exp_s / np.sum(exp_s, axis=0, keepdims=True)
    return P


def compute_loss(P: np.ndarray, y: np.ndarray) -> float:
    n = P.shape[1]
    probs = P[y, np.arange(n)]
    return float(np.mean(-np.log(np.clip(probs, 1e-15, None))))


def compute_cost(
    P: np.ndarray, y: np.ndarray, network: Dict[str, np.ndarray], lam: float
) -> float:
    loss = compute_loss(P, y)
    reg = lam * np.sum(network["W"] ** 2)
    return float(loss + reg)


def compute_accuracy(P: np.ndarray, y: np.ndarray) -> float:
    preds = np.argmax(P, axis=0)
    return float(np.mean(preds == y))


def backward_pass(
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    network: Dict[str, np.ndarray],
    lam: float,
) -> Dict[str, np.ndarray]:
    n = X.shape[1]
    G = P - Y
    grad_W = (G @ X.T) / n + 2.0 * lam * network["W"]
    grad_b = np.mean(G, axis=1, keepdims=True)
    return {"W": grad_W, "b": grad_b}


def mini_batch_gd(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    y_val: np.ndarray,
    GDparams: Dict[str, float],
    init_net: Dict[str, np.ndarray],
    lam: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    n_batch = int(GDparams["n_batch"])
    eta = float(GDparams["eta"])
    n_epochs = int(GDparams["n_epochs"])

    net = copy.deepcopy(init_net)
    n = X_train.shape[1]
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_cost": [],
        "val_cost": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X_shuf = X_train[:, perm]
        Y_shuf = Y_train[:, perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            X_batch = X_shuf[:, j_start:j_end]
            Y_batch = Y_shuf[:, j_start:j_end]
            P_batch = apply_network(X_batch, net)
            grads = backward_pass(X_batch, Y_batch, P_batch, net, lam)
            net["W"] -= eta * grads["W"]
            net["b"] -= eta * grads["b"]

        P_train = apply_network(X_train, net)
        P_val = apply_network(X_val, net)
        history["train_loss"].append(compute_loss(P_train, y_train))
        history["val_loss"].append(compute_loss(P_val, y_val))
        history["train_cost"].append(compute_cost(P_train, y_train, net, lam))
        history["val_cost"].append(compute_cost(P_val, y_val, net, lam))
        history["train_acc"].append(compute_accuracy(P_train, y_train))
        history["val_acc"].append(compute_accuracy(P_val, y_val))
        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"train_cost={history['train_cost'][-1]:.4f} | "
            f"val_cost={history['val_cost'][-1]:.4f}"
        )

    return net, history


def gradient_check(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    train_y: np.ndarray,
    rng: np.random.Generator,
) -> None:
    try:
        from torch_gradient_computations import ComputeGradsWithTorch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Install it first to run gradient_check()."
        ) from exc

    d_small = 10
    n_small = 3
    lam = 0.1
    small_net = init_network(10, d_small, rng)
    X_small = train_X[:d_small, :n_small]
    Y_small = train_Y[:, :n_small]
    y_small = train_y[:n_small]

    P = apply_network(X_small, small_net)
    my_grads = backward_pass(X_small, Y_small, P, small_net, lam)
    torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net, lam)

    eps = 1e-12
    rel_W = np.max(
        np.abs(my_grads["W"] - torch_grads["W"])
        / np.maximum(eps, np.abs(my_grads["W"]) + np.abs(torch_grads["W"]))
    )
    rel_b = np.max(
        np.abs(my_grads["b"] - torch_grads["b"])
        / np.maximum(eps, np.abs(my_grads["b"]) + np.abs(torch_grads["b"]))
    )
    print(f"Gradient check max relative error: W={rel_W:.3e}, b={rel_b:.3e}")


def plot_training_curves(history: Dict[str, List[float]], tag: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.plot(epochs, history["train_cost"], "--", label="train cost")
    plt.plot(epochs, history["val_cost"], "--", label="val cost")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title(f"Training Curves ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"curves_{tag}.png", dpi=150)
    plt.close()


def visualize_weights(network: Dict[str, np.ndarray], tag: str) -> None:
    Ws = network["W"].T.reshape((32, 32, 3, 10), order="F")
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    fig, axs = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        ax = axs[i // 5, i % 5]
        w_im = W_im[:, :, :, i]
        w_im = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im) + 1e-12)
        ax.imshow(w_im)
        ax.set_title(f"class {i}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"weights_{tag}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    cifar_dir = "Datasets/cifar-10-batches-py"
    if not os.path.isdir(cifar_dir):
        raise FileNotFoundError(
            f"Cannot find '{cifar_dir}'. Please download/extract CIFAR-10 first."
        )

    data = load_all_data(cifar_dir)
    print("Loaded and normalized CIFAR-10 splits.")

    rng = np.random.default_rng(42)
    gradient_check(data["train_X"], data["train_Y"], data["train_y"], rng)

    K, d = 10, data["train_X"].shape[0]
    configs = [
        {"lam": 0.0, "n_epochs": 40, "n_batch": 100, "eta": 0.1},
        {"lam": 0.0, "n_epochs": 40, "n_batch": 100, "eta": 0.001},
        {"lam": 0.1, "n_epochs": 40, "n_batch": 100, "eta": 0.001},
        {"lam": 1.0, "n_epochs": 40, "n_batch": 100, "eta": 0.001},
    ]

    for cfg in configs:
        init_net = init_network(K, d, rng)
        GDparams = {
            "n_batch": cfg["n_batch"],
            "eta": cfg["eta"],
            "n_epochs": cfg["n_epochs"],
        }
        tag = figure_tag_config(cfg["lam"], cfg["eta"])
        print(f"\nRunning config: {cfg}")
        trained_net, history = mini_batch_gd(
            data["train_X"],
            data["train_Y"],
            data["train_y"],
            data["val_X"],
            data["val_Y"],
            data["val_y"],
            GDparams,
            init_net,
            cfg["lam"],
            rng,
        )
        P_test = apply_network(data["test_X"], trained_net)
        test_acc = compute_accuracy(P_test, data["test_y"])
        print(f"Final test accuracy ({tag}): {test_acc * 100:.2f}%")
        plot_training_curves(history, tag)
        visualize_weights(trained_net, tag)
