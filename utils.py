import torch

def symmetric_normalize(A):
    """
    A: batches of adjacency matrices of Interactions
       size of A: [batch_size, num_nodes, num_nodes]
    """
    D_values = A.sum(-1) #Degree values; size: [batch_size, num_nodes]
    D_values_p = torch.pow(D_values, -0.5)
    D_p = torch.diag_embed(D_values_p) #size: [batch_size, num_nodes, num_nodes]
    return torch.matmul(D_p, torch.matmul(A, D_p))    
