# install pytorch 
# conda install -c pytorch pytorch
# pip3 install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio===0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

import torch
import numpy as np
from prettytable import PrettyTable
from itertools import combinations as comb
from node import get_node_data
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## the code is a combination between:
## - this forum answer: https://discuss.pytorch.org/t/maximum-mean-discrepancy-mmd-and-radial-basis-function-rbf/1875/2
## - this tutorial: https://www.kaggle.com/onurtunali/maximum-mean-discrepancy

def MMD(x, y, kernel, kernel_bandwidth):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    
    ## torch.mm performs matrix multiplication
    ## x.t() performs a transpose
    ## xx.diag() contains the square of each row/point in x
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    
    ## calculating the first summation of parts A and C
    # unsqueze adds another dimension to the tensor (,n) to (1,n)
    # expand_as duplicates the array to as many rows to match the array given 
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    # compute the distance matrix: (x - y)^2 = x^2 - 2*x*y + y^2
    dxx = rx.t() - 2 * xx + rx 
    dyy = ry.t() - 2 * yy + ry 
    dxy = rx.t() - 2 * xy + ry 
    
    if kernel == "multiscale":
        XX = kernel_bandwidth**2 * (kernel_bandwidth**2 + dxx)**-1
        YY = kernel_bandwidth**2 * (kernel_bandwidth**2 + dyy)**-1
        XY = kernel_bandwidth**2 * (kernel_bandwidth**2 + dxy)**-1
            
    if kernel == "rbf":
        XX = torch.exp(-0.5*dxx/kernel_bandwidth)
        YY = torch.exp(-0.5*dyy/kernel_bandwidth)
        XY = torch.exp(-0.5*dxy/kernel_bandwidth)
    
    if kernel == "linear":
        XX = kernel_bandwidth * xx
        YY = kernel_bandwidth * yy
        XY = kernel_bandwidth * xy

    m = x.size()[0]
    n = y.size()[0]
    
    return (1/(m*(m-1)) * torch.sum(XX) - 2/(m*n) * torch.sum(XY) + 1/(n*(n-1)) * torch.sum(YY)).item()

def avg_similarity_disimilarity_MMD(samples, similar_sets, dissimilar_sets, kernel, kernel_bandwidth, return_tables = True):
    # The similar set contains the nodes that are similar to each other and dissimilar to the the dissimilar set.
    # The dissimilar set is dissimilar to similar set nodes but they could be similar to each other.
    
    s = PrettyTable(['Nodes', 'Similar MMD'])
    d = PrettyTable(['Nodes', 'Dissimilar MMD'])
    
    combos = comb(range(len(similar_sets)),2)
    similar_mmds = []
    for combo in combos:
        x = similar_sets[combo[0]]
        y = similar_sets[combo[1]]
        sx = samples[x]
        sy = samples[y]
        mmd = MMD(sx,sy, kernel, kernel_bandwidth)
        similar_mmds.append(mmd)
        s.add_row([(x,y), mmd])

    ## calculate the mmd between each dissimilar set and the similar sets
    dissimilar_mmds = []
    for i in range(len(dissimilar_sets)):
        x = dissimilar_sets[i]
        sx = samples[x]
        for j in range(len(similar_sets)):
            y = similar_sets[j]
            sy = samples[y]
            mmd = MMD(sx,sy, kernel, kernel_bandwidth)
            if mmd > np.mean(similar_mmds):
                dissimilar_mmds.append(mmd)
                d.add_row([(x,y), mmd])
            else: 
                similar_mmds.append(mmd)
                s.add_row([(x,y), mmd])
    
    ## calculate the mmd between the dissimilar sets as they could be dissimilar/similar to each other
    if len(dissimilar_sets) > 1:
        combos = comb(range(len(dissimilar_sets)),2)
        for combo in combos:
            x = dissimilar_sets[combo[0]]
            y = dissimilar_sets[combo[1]]
            sx = samples[x]
            sy = samples[y]
            mmd = MMD(sx,sy, kernel, kernel_bandwidth)
            if mmd > np.mean(similar_mmds):
                dissimilar_mmds.append(mmd)
                d.add_row([(x,y), mmd])
            else: 
                similar_mmds.append(mmd)
                s.add_row([(x,y), mmd])

    if return_tables:
        return np.mean(similar_mmds), np.mean(dissimilar_mmds), s, d
    else:
        return np.mean(similar_mmds), np.mean(dissimilar_mmds)

def to_tensor(data):
    return torch.tensor(data).to(device)

def get_tensor_sample(data, sample_size):
    indices = get_sample_indices(data.shape[0], sample_size)
    return to_tensor(data[indices])

def get_sample_indices(range_of_indices, sample_size):
    return np.random.choice(range_of_indices, sample_size, replace=False)

def get_tensor_samples(data, experiment, sample_size, standardised = False):
    node_data = [node_data.values.astype(np.float32) for node_data in get_node_data(data, experiment)]
    
    if standardised:
        scaler = StandardScaler()
        scaler.fit(np.concatenate(node_data))
        standardised_node_data = [scaler.transform(node_data[i]) for i in range(4)]
        
    samples = {}
    for i in range(4):
        indices = get_sample_indices(node_data[i].shape[0], sample_size)
        s = to_tensor(node_data[i][indices])
        samples["pi"+str(i+2)] = s
        
        if standardised:
            stds = to_tensor(standardised_node_data[i][indices])
            samples["pi"+str(i+2)+"std"] = stds 
    
    return samples