import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import grangercausalitytests
import tslearn.metrics


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




"""
GMM for distances
"""
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



"""
Granger Causality
"""
def compute_granger_p(example):
    n_atoms = example.shape[0]
    n_timesteps = example.shape[1]
    locs = example[:,:,:] #shape: [n_atoms, n_timesteps, n_in]
    locs_re = locs.reshape(n_atoms, -1) #shape: [n_atoms, n_timesteps*2]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    #shape: [n_edges, n_nodes]
    senders = np.matmul(rel_send, locs_re) #shape: [n_edges, n_timesteps*2]
    receivers = np.matmul(rel_rec, locs_re)
    senders = senders.reshape(senders.shape[0], n_timesteps, -1)
    receivers = receivers.reshape(receivers.shape[0], n_timesteps, -1)
    #shape: [n_edges, n_timesteps, 2]
    senders = np.sqrt((senders**2).sum(-1))
    receivers = np.sqrt((receivers**2).sum(-1))
    #shape: [n_edges, n_timesteps]
    senders = senders.reshape(senders.shape[0], n_timesteps, -1)
    receivers = receivers.reshape(receivers.shape[0], n_timesteps, -1)
    #shape: [n_edges, n_timesteps, 1]

    ps = []

    for i in range(senders.shape[0]):
        result_sr = grangercausalitytests(np.concatenate([senders[i], receivers[i]], axis=-1), maxlag=4, verbose=0)
        p_sr = np.array([lag[0]["ssr_ftest"][1] for lag in result_sr.values()]).mean()
        result_rs = grangercausalitytests(np.concatenate([receivers[i], senders[i]], axis=-1), maxlag=4, verbose=0)
        p_rs = np.array([lag[0]["ssr_ftest"][1] for lag in result_rs.values()]).mean()
        ps.append(max(p_sr, p_rs))
    
    ps = np.array(ps) #shape: [n_atoms*(n_atoms-1)]
    
    return ps 


def compute_granger_sim(example):
    ps = compute_granger_p(example)
    return 1-ps




"""
DTW Distances
"""
# compute DTW distances
def compute_dtw_dist(example):
    """
    args:
      example, shape: [n_atoms, n_timesteps, n_in]
    """
    n_atoms = example.shape[0]
    n_timesteps = example.shape[1]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    #shape: [n_edges, n_atoms]
    example_re = example.reshape(example.shape[0], -1)
    #shape: [n_atoms, n_timesteps*n_in]
    senders = np.matmul(rel_send, example_re)
    receivers = np.matmul(rel_rec, example_re)
    #shape: [n_edges, n_timesteps*n_in]
    
    senders = senders.reshape(senders.shape[0], n_timesteps, -1)
    receivers = receivers.reshape(receivers.shape[0], n_timesteps, -1)
    #shape: [n_edges, n_timesteps, n_in]
    
    n_edges = n_atoms*(n_atoms-1)
    distances = []
    for i in range(n_edges):
        distances.append(tslearn.metrics.dtw(senders[i], receivers[i]))
    
    return np.array(distances) #shape: [n_edges]




#compute DTW similarity
def compute_dtw_sim(example):
    """
    args:
        example, shape:[n_atoms, n_timesteps, n_in]
    """
    distances = compute_dtw_dist(example)
    
    return np.exp(-distances)





"""
Compute heatmap
"""

def build_ground(examples_train):
    """
    build ground for heatmap based on training examples
    args:
      examples_train: training examples
    """
    max_train_x = -np.inf
    min_train_x = np.inf
    max_train_y = -np.inf
    min_train_y = np.inf
    
    for example in examples_train:
        max_example_x = example[:,:,0].max()
        max_example_y = example[:,:,1].max()
        min_example_x = example[:,:,0].min()
        min_example_y = example[:,:,1].min()
        if max_example_x > max_train_x:
            max_train_x = max_example_x
        if max_example_y > max_train_y:
            max_train_y  = max_example_y
        if min_example_x < min_train_x:
            min_train_x = min_example_x
        if min_example_y < min_train_y:
            min_train_y = min_example_y
            
    Rs = np.arange(int(min_train_x)-2.5, int(max_train_x)+2.5, 1)
    Cs = np.arange(int(min_train_y)-2.5, int(max_train_y)+2.5, 1) 
    
    #build ground
    ground = np.zeros((Rs.shape[0], Cs.shape[0], 2))   
    for i in range(Rs.shape[0]):
        for j in range(Cs.shape[0]):
            ground[i,j,0] = Rs[i]
            ground[i,j,1] = Cs[j]
        
    return ground
    

def compute_heatmap_traj(traj, ground):
    """
    compute heatmap of one trajectory
    args:
      traj: trajectory of one agent
        shape: [n_timesteps, 2]
    ground:
       coordinates of heat on ground
    """
    heatmap = np.zeros((ground.shape[0], ground.shape[1]))
    for loc in traj:
        for r in range(ground.shape[0]):
            for c in range(ground.shape[1]):
                if np.sqrt(((ground[r,c]-loc)**2).sum())< 1.:
                    heatmap[r, c] = 1.
    
    return heatmap



def compute_heatmap_sim(example, ground):
    """
    compute heatmap similarities
    args:
      example: [n_atoms, n_timesteps, n_in]
      ground: [R X C]
    """
    locs = example[:,:,:2]#extract locations
    #shape: [n_atoms, n_timesteps, 2]
    n_atoms = example.shape[0]
    n_timesteps = example.shape[1]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    locs_re = locs.reshape(locs.shape[0], -1)
    #shape: [n_atoms, n_timesteps*2]
    senders = np.matmul(rel_send, locs_re)
    receivers = np.matmul(rel_rec, locs_re)
    #shape: [n_edges, n_timesteps*2]
    senders = senders.reshape(senders.shape[0], n_timesteps, -1)
    receivers = receivers.reshape(receivers.shape[0], n_timesteps, -1)
    #shape: [n_edges, n_timesteps, 2]
    n_edges = n_atoms*(n_atoms-1)
    sims = []
    for i in range(n_edges):
        traj_s = senders[i]
        traj_r = receivers[i]
        heatmap_s = compute_heatmap_traj(traj_s, ground)
        heatmap_r = compute_heatmap_traj(traj_r, ground)
        sim_sr = (heatmap_r*heatmap_s).sum()/(np.sqrt(ground.shape[0]*ground.shape[1]))
        sims.append(sim_sr)
        
    return np.array(sims) #shape: [n_edges]





"""
compute pairwise similarity for examples

"""
def compute_sims(example, ground):
    """
    compute pairwise similarity and dissimilarity for one example
    args:
      example: [n_atoms, n_timesteps, n_in]
    """
    #compute GMM distances Sim
    gmm_sim = compute_gmmDist_example(example)
    gmm_dissim = (1-gmm_sim).reshape(-1,1)
    gmm_sim = gmm_sim.reshape(-1,1)
    
    #compute Granger Causality Sim
    #granger_sim = compute_granger_sim(example)
    #granger_dissim = (1-granger_sim).reshape(-1,1)
    #granger_sim = granger_sim.reshape(-1,1)
    
    #compute DTW sim
    dtw_sim = compute_dtw_sim(example)
    dtw_dissim = (1-dtw_sim).reshape(-1,1)
    dtw_sim = dtw_sim.reshape(-1, 1)
    
    #compute heatmap sim
    heatmap_sim = compute_heatmap_sim(example, ground)
    heatmap_dissim = (1-heatmap_sim).reshape(-1,1)
    heatmap_sim = heatmap_sim.reshape(-1, 1)
    
    sim = np.concatenate([gmm_sim, dtw_sim, heatmap_sim], axis=-1)
    dissim = np.concatenate([gmm_dissim, dtw_dissim, heatmap_dissim], axis=-1)
    
    return sim, dissim



























