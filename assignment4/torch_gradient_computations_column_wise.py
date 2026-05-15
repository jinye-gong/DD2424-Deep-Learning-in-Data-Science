import numpy as np
import torch

# assumes X has size d x tau, h0 has size m x 1, etc
def ComputeGradsWithTorch(X, y, h0, RNN):

    tau = X.shape[1]

    Xt = torch.from_numpy(X)
    ht = torch.from_numpy(h0)

    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)


    ## give informative names to these torch classes        
    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=0) 
    
    # create an empty tensor to store the hidden vector at each timestep
    Hs = torch.empty(h0.shape[0], X.shape[1], dtype=torch.float64)
    
    hprev = ht
    for t in range(tau):

        #### BEGIN your code ######

        a = torch.matmul(torch_network['W'], hprev) + torch.matmul(torch_network['U'], Xt[:, t:t+1]) + torch_network['b']
        h = apply_tanh(a)
        Hs[:, t:t+1] = h
        hprev = h

        #### END of your code ######            

    Os = torch.matmul(torch_network['V'], Hs) + torch_network['c']        
    P = apply_softmax(Os)    
    
    # compute the loss
    
    loss = torch.mean(-torch.log(P[y, np.arange(tau)]))
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays
    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads
