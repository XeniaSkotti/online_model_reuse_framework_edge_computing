# use install pytorch
# conda install -c pytorch pytorch
# pip3 install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio===0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

import torch
import numpy as np
from prettytable import PrettyTable
from itertools import combinations as comb

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
        kernel: kernel type such as "multiscale", "rbf", "linear"
        kernel_bandwidth: specifies the scalar value to be used by the kernel
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
    if min(m,n) < 1:
        print(f"({m},{n})", end = " ")
    
    return (1/(m*(m-1)) * torch.sum(XX) - 2/(m*n) * torch.sum(XY) + 1/(n*(n-1)) * torch.sum(YY)).item()

def avg_similarity_disimilarity_MMD(samples, similar_nodes, other_nodes, 
                                    kernel, kernel_bandwidth, return_tables = True):
    
    """ Calculates the average similarity and dissimilarity MMD (ASDMMD) between the 
    given nodes. 
    
    Args: 
        samples: dictionary associating each node (pi2-pi5) with a sample used for the 
                 MMD calculation.
        similar_nodes: nodes which we have visually identified as similar to each other.
        other_nodes: the rest of the nodes which we're unsure of.
        kernel: the kernel type to be used for the MMD calculation.
        kernel_bandwidth: scalar value to be used by the kernel in the MMD calculation.
        return_tables: boolean value which determines whether we return the tables s(imilar)
                       and d(issimilar) containing a pair of nodes and their corresponding 
                       MMD.  
    
    We use the similar_nodes to calculate a baseline average similarity MMD (ASMMD). 
    Then, we use ASMMD to judge whether the other_nodes are similar to each other or 
    to the similar_nodes. If they are we calculate the new ASMMD and we use this to 
    judge the next pair. Otherwise, we use the calculated MMD value in the 
    average dissimilar MMD (ADMMD).
    
    Regardless of whether we return the tables s and d we always return the ASMMD.
    
    """
    
    ## The tables are constructed for experimentation purposes to understand 
    ## relationships between pairs and pin point which pairs are similar. 
    
    s = PrettyTable(['Nodes', 'Similar MMD'])
    d = PrettyTable(['Nodes', 'Dissimilar MMD'])
    
    ## Calculating the baseline ASMMD
    combos = comb(range(len(similar_nodes)),2)
    similar_mmds = []
    for combo in combos:
        x = similar_nodes[combo[0]]
        y = similar_nodes[combo[1]]
        sx = samples[x]
        sy = samples[y]
        mmd = MMD(sx,sy, kernel, kernel_bandwidth)
        similar_mmds.append(mmd)
        s.add_row([(x,y), mmd])

    ## Comparing the other_nodes with each of the similar_nodes
    dissimilar_mmds = []
    for i in range(len(other_nodes)):
        x = other_nodes[i]
        sx = samples[x]
        for j in range(len(similar_nodes)):
            y = similar_nodes[j]
            sy = samples[y]
            mmd = MMD(sx,sy, kernel, kernel_bandwidth)
            ## allow for the mmd to be 5% higher than the current ASMMD
            if mmd > np.mean(similar_mmds) + np.mean(similar_mmds) * 0.05:
                dissimilar_mmds.append(mmd)
                d.add_row([(x,y), mmd])
            else: 
                similar_mmds.append(mmd)
                s.add_row([(x,y), mmd])
    
    ## Calculating the MMD between each 
    if len(other_nodes) > 1:
        combos = comb(range(len(other_nodes)),2)
        for combo in combos:
            x = other_nodes[combo[0]]
            y = other_nodes[combo[1]]
            sx = samples[x]
            sy = samples[y]
            mmd = MMD(sx,sy, kernel, kernel_bandwidth)
            if mmd > np.mean(similar_mmds) + np.mean(similar_mmds) * 0.05:
                dissimilar_mmds.append(mmd)
                d.add_row([(x,y), mmd])
            else: 
                similar_mmds.append(mmd)
                s.add_row([(x,y), mmd])

    if return_tables:
        return np.mean(similar_mmds), np.mean(dissimilar_mmds), s, d
    else:
        return np.mean(similar_mmds)

def to_tensor(data):
    return torch.tensor(data).to(device)

def get_tensor_sample(data, sample_size):
    indices = get_sample_indices(data.shape[0], sample_size)
    return to_tensor(data[indices])

def get_sample_indices(range_of_indices, sample_size):
    return np.random.choice(range_of_indices, sample_size, replace=False)

def get_tensor_samples(node_data, sample_size):        
    samples = {}
    for i in range(len(node_data)):
        indices = get_sample_indices(node_data[i].shape[0], sample_size)
        if "humidity" in node_data[i].columns:
            s = node_data[i][["humidity", "temperature"]]
            samples["pi"+str(i+2)] = to_tensor(s.values.astype(np.float32)[indices])
        else:
            s = node_data[i][["x", "y", "z"]]
            samples["pi"+str(i+1)] = to_tensor(s.values.astype(np.float32)[indices])
    
    return samples