import torch
from graphgym.config import cfg

def node_degree(edge_index, n=None, mode='in'):
    # edge_index[0] is source and edge_index[1] is target.
    if mode == 'in':
        index = edge_index[1, :]
    elif mode == 'out':
        index = edge_index[0, :]
    else:
        index = edge_index.flatten()
    # n = edge_index.max() + 1 if n is None else n
    if cfg.dataset.name == 'cannes.txt':
        n = 438537
        
    # n=38918
    # n=116408
    # n= 256491
    degree = torch.zeros(n) # 所有节点的度张量
    ones = torch.ones(index.shape[0]) # 该时间步内的节点
    return degree.scatter_add_(0, index, ones)

'''def node_degree(edge_index, n=None, mode='in'):
    if mode == 'in':
        index = edge_index[1, :]
    elif mode == 'out':
        index = edge_index[0, :]
    else:
        index = edge_index.flatten()

    n=35768

    # If num_nodes is not provided, infer it from the edge_index.
    """if num_nodes is None:
        num_nodes = int(torch.max(edge_index)) + 1"""

    # Initialize a tensor to store node degrees.
    degree = torch.bincount(index, minlength=n)

    return degree'''

# edge_index = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
# print(compute_degree(edge_index, mode='in'))
# print(compute_degree(edge_index, mode='out'))
# print(compute_degree(edge_index, mode='both'))
