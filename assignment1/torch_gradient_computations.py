import torch
import numpy as np


def ComputeGradsWithTorch(X, y, network_params, lam=0.0):

    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)

    # will be computing the gradient w.r.t. these parameters
    W = torch.tensor(network_params['W'], requires_grad=True)
    b = torch.tensor(network_params['b'], requires_grad=True)    
    
    N = X.shape[1]
    
    scores = torch.matmul(W, Xt)  + b;

    ## give an informative name to this torch class
    apply_softmax = torch.nn.Softmax(dim=0)

    # apply softmax to each column of scores
    P = apply_softmax(scores)
    
    ## compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))
    cost = loss + lam * torch.sum(torch.multiply(W, W))

    # compute the backward pass relative to the loss and the named parameters 
    cost.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()

    return grads


def ComputeGradsSigmoidMultiBCE(X, y, network_params, lam=0.0):
    """Multi-class independent sigmoid + mean multi-label BCE (assignment bonus)."""
    Xt = torch.from_numpy(X)
    N = X.shape[1]
    K = network_params["W"].shape[0]
    W = torch.tensor(network_params["W"], requires_grad=True)
    b = torch.tensor(network_params["b"], requires_grad=True)
    scores = torch.matmul(W, Xt) + b
    P = torch.sigmoid(scores)
    y_idx = torch.from_numpy(np.asarray(y, dtype=np.int64))
    Y = torch.zeros(K, N, dtype=P.dtype)
    Y[y_idx, torch.arange(N, dtype=torch.long)] = 1.0
    eps = 1e-15
    Pc = P.clamp(eps, 1.0 - eps)
    term = (1.0 - Y) * torch.log(1.0 - Pc) + Y * torch.log(Pc)
    loss = torch.mean(-torch.sum(term, dim=0) / K)
    cost = loss + lam * torch.sum(W * W)
    cost.backward()
    return {"W": W.grad.numpy(), "b": b.grad.numpy()}
