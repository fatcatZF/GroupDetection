import numpy as np
from scipy.stats import norm


#Gaussian Mixture Models
N0 = norm(0, 0.5)
N1 = norm(0, 1.2)
N2 = norm(0, 3.7)
N3 = norm(0, 7.6)

GMM = [N0, N1, N2, N3]



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def create_edgeNode_relation(num_nodes, self_loops=False):
    if self_loops:
        indices = np.ones([num_nodes, num_nodes])
    else:
        indices = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rel_rec = np.array(encode_onehot(np.where(indices)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(indices)[1]), dtype=np.float32)
    
    return rel_rec, rel_send 



def compute_gmm(distance, GMM=GMM):
    """
    args:
        distance:
            distance between 2 agents at one timestep
        GMM:
            Gaussian Mixture Model
    """
    num = len(GMM)
    probs = [N.pdf(distance) for N in GMM]
    return sum(probs)/num



def compute_gmmDist_example(example):
    """
    args:
      example, shape: [n_atoms, n_timesteps, n_in]
    """
    #extract locations of the example
    locs = example[:,:,:2] #extract locations, shape: [n_atoms, n_timesteps, 2]
    n_atoms = example.shape[0]
    n_timesteps = example.shape[1]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    #shape: [n_edges, n_atoms]
    locs_re = locs.reshape(locs.shape[0], -1) #shape: [n_atoms, n_timesteps*2]
    senders_locs = np.matmul(rel_send, locs_re) #shape: [n_edges, n_timesteps*2]
    receivers_locs = np.matmul(rel_rec, locs_re)
    senders_locs = senders_locs.reshape(senders_locs.shape[0], n_timesteps, -1)
    receivers_locs = receivers_locs.reshape(receivers_locs.shape[0], n_timesteps, -1)
    distances = np.sqrt(((senders_locs-receivers_locs)**2).sum(-1)) #shape: [n_edges, n_timesteps]
    #compute GMM probs
    distances_re = distances.reshape(-1)
    probs = np.array([compute_gmm(dist) for dist in distances_re])
    probs = probs.reshape(distances.shape[0], -1)
    probs = probs.mean(-1) #shape: [n_atoms*(n_atoms-1)]
    
    return probs









