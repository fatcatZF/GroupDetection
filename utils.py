import torch

def symmetrize(A):
    """
    args:
        A: batches of Interaction matrices
    return: A_sym: symmetric version of A
    """
    AT = A.transpose(-1,-2)
    return 0.5*(A+AT)


def symmetric_normalize(A):
    """
    args:
      A: batches of adjacency matrices of Interactions
         size of A: [batch_size, num_nodes, num_nodes]       
   return: A_norm = (D**-0.5)A(D**-0.5), where D is the diagonal matrix of A
    """
    D_values = A.sum(-1) #Degree values; size: [batch_size, num_nodes]
    D_values_p = torch.pow(D_values, -0.5)
    D_p = torch.diag_embed(D_values_p) #size: [batch_size, num_nodes, num_nodes]
    return torch.matmul(D_p, torch.matmul(A, D_p)) 


