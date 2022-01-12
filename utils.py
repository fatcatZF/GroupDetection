import numpy as np
import torch

def symmetrize(A):
    """
    args:
        A: batches of Interaction matrices
    return: A_sym: symmetric version of A
    """
    AT = A.transpose(-1,-2)
    return 0.5*(A+AT)


def laplacian_smooth(A):
    """
    args:
      A: batches of adjacency matrices of symmetric Interactions
         size of A: [batch_size, num_edgeTypes, num_nodes, num_nodes]       
    return: A_norm = (D**-0.5)A(D**-0.5), where D is the diagonal matrix of A
    """
    I = torch.eye(A.size(-1))
    I = I.unsqueeze(0).unsqueeze(1)
    I = I.expand(A.size(0), A.size(1), I.size(2), I.size(3))
    #size: [batch_size, num_edgeTypes, num_atoms, num_atoms]
    A_p = A+I
    D_values = A_p.sum(-1) #Degree values; size: [batch_size, num_nodes]
    D_values_p = torch.pow(D_values, -0.5)
    D_p = torch.diag_embed(D_values_p) #size: [batch_size, num_nodes, num_nodes]
    return torch.matmul(D_p, torch.matmul(A_p, D_p)) 


def laplacian_sharpen(A):
    """
    args:
        A; batches of adjacency matrices corresponding to edge types
          size: [batch_size, num_edgeTypes, num_nodes, num_nodes]
    """
    I = torch.eye(A.size(-1))
    I = I.unsqueeze(0).unsqueeze(1)
    I = I.expand(A.size(0), A.size(1), I.size(2), I.size(3))
    #size: [batch_size, num_edgeTypes, num_atoms, num_atoms]
    Ap = 2*I-A
    D_values = A.sum(-1)+2 #shape: [batch_size, num_edgeTypes, num_atoms]
    D_values_p = torch.pow(D_values, -0.5)
    D_p = torch.diag_embed(D_values_p)
    
    return torch.matmul(D_p, torch.matmul(Ap, D_p)) 
    
    
    
    



def nll_gaussian(preds, target, variance, add_const=False):
    """
    loss function
    copied from https://github.com/ethanfetaya/NRI/blob/master/utils.py
    """
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))
    

def get_noise(shape, noise_type="gaussian"):
    """copied from https://github.com/huang-xx/STGAT/blob/master/STGAT/models.py"""
    if noise_type == "gaussian":
        return torch.randn(*shape)
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    #not tested yet
    kl_div = preds*(torch.log(preds+eps)-log_prior)
    return kl_div.sum()/(num_atoms*preds.size(0))



def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds*torch.log(preds+eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum()/(num_atoms*preds.size(0))


def kl_gaussian(mu, sigma):
    return ((0.5*(1+torch.log(sigma**2)-mu**2-sigma**2)).sum()/(mu.size(0)*mu.size(1)))


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices

def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices

def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices

def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()

def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()



def edge_accuracy(preds, target):
    """compute pairwise group accuracy"""
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))



