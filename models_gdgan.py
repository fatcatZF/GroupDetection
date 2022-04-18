"""
GDGAN models
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import math




class LSTMCell(nn.Module):
    def __init__(self, n_in, n_hid):
        """
        args:
          n_in: dimensions of input features
          n_hid: hidden dimensions
        """
        super(LSTMCell, self).__init__()
        self.lstm_cell = nn.LSTMCell(n_in, n_hid)
        self.n_hid = n_hid
        
    def forward(self, inputs, hc=None):
        """
        args:
          inputs, shape: [batch_size, num_atoms, num_features]
          hc, a tuple include hidden state and cell state
        """
        x = inputs.view(inputs.size(0)*inputs.size(1),-1)
        #shape: [total_atoms, num_features]
        if hc is None:
            hidden = torch.zeros(x.size(0),self.n_hid)
            cell = torch.zeros_like(hidden)
            if inputs.is_cuda:
                hidden = hidden.cuda()
                cell = cell.cuda()
            hc = (hidden, cell)
        h, c = self.lstm_cell(x, hc)
        #shape: [batch_size*n_atoms, n_hid]
    
        return h, c
    
    
   
class LSTMEncoder(nn.Module):
    """LSTM Encoder"""
    def __init__(self, n_in, n_emb=16, n_h=32):
        super(LSTMEncoder, self).__init__()
        self.fc_emb = nn.Linear(n_in, 2*n_emb)
        self.lstm_cell = LSTMCell(2*n_emb, n_h)
        
    def forward(self, inputs, rel_rec=None, rel_send=None):
        """
        args:
            inputs: [batch_size, num_atoms, num_timesteps, n_in]
            
        return: latents of trajectories of atoms
        """
        batch_size = inputs.size(0)
        num_atoms = inputs.size(1)
        num_timesteps = inputs.size(2)
        hc = None
        hs = []
        for i in range(num_timesteps):
            inputs_i = inputs[:,:,i,:]
            #shape: [batch_size, n_atoms, n_in]
            inputs_i = self.fc_emb(inputs_i)
            #shape: [batch_size, n_atoms, n_emb]
            h,c = self.lstm_cell(inputs_i, hc)
            #shape: h:[batch_size*n_atoms, n_h]
            h_re = h.view(batch_size, num_atoms, -1)
            hs.append(h_re)
            hc = (h,c)
        hs = torch.stack(hs) #shape: [n_timesteps, batch_size, n_atoms, n_h]
        hs = torch.permute(hs, (1,2,0,-1))
        return hs
    



class SoftAttention(nn.Module):
    """Soft Attention"""
    def __init__(self, n_h=32):
        super(SoftAttention, self).__init__()
        self.soft_att = nn.Linear(2*n_h, 1)
        
    def forward(self, inputs, rel_rec_t, rel_send_t):
        """
        inputs: [batch_size, n_atoms, n_timesteps, n_h]
        rel_rec_t: [n_timesteps*n_timesteps, n_timesteps]
        rel_send_t: [n_timesteps*n_timesteps, n_timesteps]
        """
        batch_size = inputs.size(0)
        n_atoms = inputs.size(1)
        n_timesteps = inputs.size(2)
        hs = inputs.view(batch_size*n_atoms, n_timesteps, -1)
        #shape: [batch_size*n_atoms, n_timesteps, n_h]
        senders = torch.matmul(rel_send_t, hs)
        #shape: [batch_size*n_atoms, n_timesteps**2, n_h]
        receivers = torch.matmul(rel_rec_t, hs)
        #shape: [batch_size*n_atoms, n_timesteps**2, n_h]
        edges = torch.cat([senders, receivers], dim=-1)
        #shape: [batch_size*n_atoms, n_timesteps**2, 2*n_h]
        scores = self.soft_att(edges) 
        #shape: [batch_size*n_atoms, n_timesteps**2, 1]
        scores = scores.squeeze(-1)
        #shape: [batch_size*n_atoms, n_timesteps**2]
        scores_diag = torch.diag_embed(scores)
        #shape: [batch_size*n_atoms, n_timesteps**2, n_timesteps**2]
        adj = torch.matmul(rel_send_t.t(), torch.matmul(scores_diag, rel_rec_t))
        #shape: [batch_size*n_atoms, n_timesteps, n_timesteps]
        adj_normalized = F.softmax(adj, dim=-1)
        cs = torch.matmul(adj_normalized, hs)
        #shape: [batch_size*n_atoms, n_timesteps, n_h]
        cs = cs.view(batch_size, n_atoms, n_timesteps, -1)
        #Soft Attention Context: [batch_size, n_atoms, n_timesteps, n_h]
        
        return cs
    
    
class HardwiredAttention(nn.Module):
    """Hardwired Attention"""
    def __init__(self):
        super(HardwiredAttention, self).__init__()
        
    def forward(self, locs, hidden, rel_rec, rel_send, eps=1e-5):
        """
        locs: locations, shape: [batch_size, n_atoms, n_timesteps, 2]
        hidden, shape: [batch_size, n_atoms, n_timesteps, n_h]
        rel_rec,rel_send; shape: [n_atoms*(n_atoms-1), n_atoms]
        """
        batch_size = locs.size(0)
        n_atoms = locs.size(1)
        n_timesteps = locs.size(2)
        locs_re = locs.reshape(batch_size, n_atoms, -1)
        senders = torch.matmul(rel_send, locs_re)
        receivers = torch.matmul(rel_rec, locs_re)
        #shape: [batch_size, n_atoms*(n_atoms-1), n_timesteps*2]
        senders = senders.view(batch_size, n_atoms*(n_atoms-1), n_timesteps, -1)
        receivers = receivers.view(batch_size, n_atoms*(n_atoms-1), n_timesteps, -1)
        
        distances = torch.sqrt(((senders-receivers)**2).sum(-1))
        #shape: [batch_size, n_atoms*(n_atoms-1), n_timestpes]
        weights = 1/(distances+eps)
        
        weights = weights.permute(0,2,1)
        #shape: [batch_size, n_timesteps, n_atoms*(n_atoms-1)]
        weights_diag = torch.diag_embed(weights)
        #shape: [batch_size, n_timesteps, n_edges, n_edges]
        
        adj = torch.matmul(rel_send.t(), torch.matmul(weights_diag, rel_rec))
        
        hidden_permute = hidden.permute(0,2,1,-1)
        
        ch = torch.matmul(adj, hidden_permute)
        #shape: [batch_size, n_timesteps, n_atoms, n_h]
        
        ch = ch.permute(0, 2, 1, -1)
        #shape: []
        
        return ch
        
        
        
        










    





