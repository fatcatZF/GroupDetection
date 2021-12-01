
import numpy as np
import matplotlib.pyplot as plt
import time


def dynamic_rule1(edges, age, num_atoms, spring_types, spring_prob,
                 reset_time=5):
    """
    reset the adjacency matrix every reset_time timesteps
    :param edges: current edges
    :param reset_time: time for reset the interaction edges
    :param age: the age of edges
    return: new edges
    """
    if (age[0,1])%reset_time == 0:
        edges = np.random.choice(spring_types,
                                 size=(num_atoms, num_atoms),
                                 p=spring_prob)
        edges = np.tril(edges)+np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        return edges
    
    age+=1
    np.fill_diagonal(age,0)
    
    return edges, age


def dynamic_rule2(edges, age, num_atoms, spring_types, spring_prob,
                  factor=0.001):
    """
    flip the edges depend on the age of the edges:
        flip_prob = 1-exp(-age*factor)
    :param: edges: current edges
    :param: age: ages of the edges
    :param: factor: flip probability increasing factor
    return: new edges and reset edge ages
    """
    flip_probs = 1 - np.exp(-age*factor)
    whether_flip = np.zeros_like(flip_probs)
    for i in range(whether_flip.shape[0]):
        for j in range(whether_flip.shape[1]):
            whether_flip[i,j] = np.random.choice([1,0], p = [flip_probs[i,j],1-flip_probs[i,j]])
    whether_flip = np.tril(whether_flip)+np.tril(whether_flip, -1).T
    np.fill(whether_flip, 0)
    whether_flip = whether_flip.astype("bool")
    edges[whether_flip] = (1-edges)[whether_flip]
    age += 1
    age[whether_flip] = 0
    np.fill_diagonal(age,0)
    return edges, age
    


class SpringSim(object):
    """
    copied from https://github.com/ethanfetaya/NRI/blob/master/data/synthetic_sim.py
    adapted for Dynamic Graphs
    """
    def __init__(self, n_balls=5, box_size=5., loc_std=0.5, vel_norm=0.5,
                 interaction_strength=0.1, noise_var=0., dynamic=False, dynamic_rule=None):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var
        
        self._spring_types = np.array([0., 0.5, 1.0])
        self._delta_T = 0.001
        self._max_F = 0.1/self._delta_T
        self.dynamic = dynamic and (dynamic_rule is not None)
        if self.dynamic:
            self.dynamic_rule = dynamic_rule
        
    def _energy(self, loc, vel, edges):
        with np.errstate(divide="ignore"):
            K = 0.5*(vel**2).sum() #Kinetic energy
            U = 0 #potential energy
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i!=j:
                        r = loc[:,i]-loc[:,j]
                        dist = np.sqrt((r**2).sum())
                        U += 0.5*self.interaction_strength*edges[
                            i,j]*(dist**2)/2
            return U+K
        
    
    def _clamp(self, loc, vel):
        """
        :param loc: 2xN location at one time stamp
        :param vel: 2XN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after 
          elastically colliding with walls
        """
        assert (np.all(loc<self.box_size*3))
        assert (np.all(loc>-self.box_size*3))
        
        over = loc > self.box_size
        loc[over] = 2*self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))
        
        vel[over] = -np.abs(vel[over])
        
        under = loc < -self.box_size
        loc[under] = -2*self.box_size - loc[under]
        
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])
        
        return loc, vel
    
    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matrix
        Output: dist is a NxM matrix where dist[i,j] is the square norm
                between A[i,:] and B[j,:]
        i.e. dist[i,j] = |A[i,:]-B[j,:]|^2
        """
        A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
        B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm+B_norm-2*A.dot(B.transpose())
        return dist
    
    def sample_trajectory(self, T=10000, sample_freq=100,
                          spring_prob=[0.6, 0, 0.4]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T/sample_freq-1)
        diag_mask = np.ones((n,n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        edges = np.random.choice(self._spring_types,
                                 size = (self.n_balls, self.n_balls),
                                 p=spring_prob)
        edges = np.tril(edges)+np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        sampled_indices = []
        loc = np.zeros((T_save,2,n))
        vel = np.zeros((T_save,2,n))
        loc_all = np.zeros((T,2, n))
        vel_all = np.zeros((T,2,n))
        loc_next = np.random.randn(2,n)*self.loc_std
        vel_next = np.random.randn(2,n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next =  vel_next*self.vel_norm/v_norm
        loc[0,:,:], vel[0,:,:] = self._clamp(loc_next, vel_next)
        loc_all[0,:,:], vel[0,:,:] = self._clamp(loc_next, vel_next)
        
        if self.dynamic:
            all_edges_sampled = np.zeros((T_save, n, n))
            all_edges_sampled[0,:,:] = edges
            all_edges = np.zeros((T,n,n))
            all_edges[0,:,:] = edges
            
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F
            
            vel_next += self._delta_T*F
            
            #initialize age of edges
            age = np.zeros_like(edges)
            # run leapfrog
            for i in range(1,T):
                loc_next += self._delta_T*vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)
                loc_all[0,:,:], vel[0,:,:] = loc_next, vel_next
                
                if i%sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    if self.dynamic: all_edges_sampled[counter,:,:] = edges
                    sampled_indices.append(i)
                    counter += 1
                    
                if self.dynamic:
                    edges, age = self.dynamic_rule(edges, age, self.n_balls, self._spring_types, spring_prob)
                    all_edges[i,:,:] = edges
                    
                forces_size = -self.interaction_strength*edges
                np.fill_diagonal(forces_size, 0)
                
                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                                                                       
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
                
            
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            sampled_indices = np.array(sampled_indices)
            
            if self.dynamic: 
                edges = all_edges_sampled
                return loc, vel, loc_all, vel_all ,edges, all_edges, sampled_indices
            
            return loc, vel, loc_all, vel_all, edges, sampled_indices
                
                
                
                
                    
            
                                           
        
                                 
