import numpy as np
import torch



def create_edgeNode_relation(num_nodes, self_loops=False):
    if self_loops:
        indices = np.ones([num_nodes, num_nodes])
    else:
        indices = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rel_rec = np.array(encode_onehot(np.where(indices)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(indices)[1]), dtype=np.float32)
    rel_rec = torch.from_numpy(rel_rec)
    rel_send = torch.from_numpy(rel_send)
    
    return rel_rec, rel_send






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
    return -((0.5*(1+torch.log(sigma**2)-mu**2-sigma**2)).sum()/(mu.size(0)*mu.size(1)))


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


def edge_precision(preds, target):
    """compute pairwise group/non-group recall"""
    _, preds = preds.max(-1)
    group_precision = ((target[preds==1]==1).sum())/preds[preds==1].sum()
    non_group_precision = ((target[preds==0]==0).sum())/(preds[preds==0]==0).sum()
    return group_precision.item(), non_group_precision.item()
    
    

def edge_recall(preds, target):
    """compute pairwise group/non-group precision"""
    _,preds = preds.max(-1)
    group_recall = ((preds[target==1]==1).sum())/(target[target==1]).sum()
    non_group_recall = ((preds[target==0]==0).sum())/(target[target==0]==0).sum()
    return group_recall, non_group_recall
    
    



def compute_mitre(a, b):
    """
    compute mitre 
    more details: https://aclanthology.org/M95-1005.pdf
    args:
      a,b: list of groups; e.g. a=[[1.2],[3],[4]], b=[[1,2,3],[4]]
    Return: 
      mitreLoss a_b
      
    """
    total_m = 0 #total missing links
    total_c = 0 #total correct links
    for group_a in a:
        pa = 0 #partitions of group_a in b
        part_group = []#partition group
        size_a = len(group_a) #size of group a
        for element in group_a:
            for group_b in b:
                if element in group_b:
                    if part_group==group_b:
                        continue
                    else:
                        part_group = group_b
                        pa+=1
        total_c += size_a-1
        total_m += pa-1
        
    return (total_c-total_m)/total_c




def create_counterPart(a):
    """
    add fake counter parts for each agent
    args:
      a: list of groups; e.g. a=[[0,1],[2],[3,4]]
    """
    a_p = []
    for group in a:
        if len(group)==1:#singleton
            element = group[0]
            element_counter = -(element+1)#assume element is non-negative
            new_group = [element, element_counter]
            a_p.append(new_group)
        else:
            a_p.append(group)
            for element in group:
                element_counter = -(element+1)
                a_p.append([element_counter])
    return a_p




def compute_groupMitre(target, predict):
    """
    compute group mitre
    args: 
      target,predict: list of groups; [[0,1],[2],[3,4]]
    return: recall, precision, F1
    """
    #create fake counter agents
    target_p = create_counterPart(target)
    predict_p = create_counterPart(predict)
    recall = compute_mitre(target_p, predict_p)
    precision = compute_mitre(predict_p, target_p)
    F1 = 2*recall*precision/(recall+precision)
    return recall, precision, F1


