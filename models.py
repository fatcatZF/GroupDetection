import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class MotionEmbedding(nn.Module):
    def __init__(self, n_in, n_emb):
        """
        n_in: number of features
        n_emb: embedding sizes
        """
        super(MotionEmbedding, self).__init__()
        self.fc = nn.Linear(n_in, n_emb)
        
        
    def forward(self, X):
        """
        X: states over time, shape: [batch_size, num_objects, num_timesteps, num_features]
        """
        deltaX = X[:,:,1:,:]-X[:,:,:-1,:] #Motion, shape: [batch_size, num_objects, num_timesteps-1, n_in]
        return self.fc(deltaX) #Embeddings of shape: [batch_size, num_objects, num_timesteps-1, n_emb]
        

class GRUCell(nn.Module):
    def __init__(self, n_in, n_h):
        """
        n_in: input dimensions
        n_h: hidden dimensions
        """
        super(GRUCell, self).__init__()
        self.l_ir = nn.Linear(n_in, n_h)
        self.l_hr = nn.Linear(n_h, n_h)
        self.l_iz = nn.Linear(n_in, n_h)
        self.l_hz = nn.Linear(n_h, n_h)
        self.l_in = nn.Linear(n_in, n_h)
        self.l_hn = nn.Linear(n_h,n_h)
        
    def forward(self, X, h):
        """
        X: input at time t; shape: [batch_size, num_objects, n_in]
        h: hidden state at time t-1; shape: [batch_size, num_objects, n_h]
        """
        r_t = torch.sigmoid(self.l_ir(X)+self.l_hr(h))
        z_t = torch.sigmoid(self.l_iz(X)+self.l_hz(h))
        n_t = torch.tanh(self.l_in(X)+r_t*self.l_hn(h))
        h_t = (1-z_t)*n_t+z_t*h
        return h_t
    

class LSTMCell(nn.Module):
    def __init__(self, n_in, n_h):
        """
        n_in: input dimensions
        n_h: hidden dimensions
        """
        super(LSTMCell, self).__init__()
        self.l_ii = nn.Linear(n_in, n_h)
        self.l_hi = nn.Linear(n_h, n_h)
        self.l_if = nn.Linear(n_in, n_h)
        self.l_hf = nn.Linear(n_h, n_h)
        self.l_ig = nn.Linear(n_in, n_h)
        self.l_hg = nn.Linear(n_h, n_h)
        self.l_io = nn.Linear(n_in, n_h)
        self.l_ho = nn.Linear(n_h, n_h)
        
    def forward(self, x, c, h):
        """
        x: input at time t; shape: [num_sims, num_atoms, num_features]
        h: hidden at time t-1 ; shape: [num_sims, num_atoms, hidden_size]
        c: cell at time t-1; shape: [num_sims, num_atoms, hidden_size]
        """
        i_t = torch.sigmoid(self.l_ii(x)+self.l_hi(h))
        f_t = torch.sigmoid(self.l_if(x)+self.l_hf(h))
        g_t = torch.tanh(self.l_ig(x)+self.l_hg(h))
        o_t = torch.sigmoid(self.l_io(x)+self.l_ho(h))
        c_t = f_t*c + i_t*g_t
        h_t = o_t * torch.tanh(c_t)
        return c_t, h_t
        




class RNNEncoder(nn.Module):
    pass


class RNNDecoder(nn.Module):
    pass