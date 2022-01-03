import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *



class GCNLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(n_in, n_out)        
        self.init_weights()
                
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
        
    def forward(self, A_norm, inputs=None):
        """
        A_norm: normalized adjacency matrix
           shape: [batch_size, num_atoms, num_atoms]
        inputs: attributes matrix of nodes
           shape: [batch_size, num_atoms, num_atoms]
           if inputs==None:
               only consider structure information
        """
        if inputs is None:
            I = torch.eye(A_norm.size(-1))
            inputs = I.unsqueeze(0).expand(A_norm.size(0),I.size(0),I.size(1))
        
        return torch.matmul(A_norm, self.fc(inputs))
    





class TypedGCNAutoEncoder(nn.Module):
    """2 layer Typed GCN Encoder"""
    def __init__(self, n_in, n_hid, n_out, edge_types, dropout=0.,
                 skip_first=True, mode="encoder"):
        """
        n_in: num of features of nodes
        n_hid: number of hidden dimensions
        n_out: number of output dimension
        edge_types: number of edge_types
        dropout: dropout rate when training this model
        skip_first: if true, the first edge type denotes the 
            non-interaction edge type
        mode: encoder or decoder
        """
        
        super(TypedGCNAutoEncoder, self).__init__()
        #first layer of GCNs
        self.gcns_1 = nn.ModuleList(
            [GCNLayer(n_in, n_hid) for _ in range(edge_types)])
        self.gcns_2 = nn.ModuleList(
            [GCNLayer(n_hid, n_out) for _ in range(edge_types)])
        
        self.dropout = dropout
        self.mode = mode
        self.skip_first = skip_first
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        
    def forward(self, A_norm, inputs=None):
        """
        A_norm: normalized adjacency matrix
            shape: [batch_size, num_edgetypes, num_atoms, num_atoms]
        inputs: node attributes
             shape: [batch_size, num_atoms, n_dimension]
             if None, Identity matrix will be used:
                 [batch_size, num_atoms, num_atoms]
        """
        if inputs is None:
            I = torch.eye(A_norm.size(-1))
            inputs = I.unsqueeze(0).expand(A_norm.size(0),I.size(0),I.size(1))
            
        if self.skip_first:
            start_idx = 1
        else:
            start_idx = 0
        
            
        all_outputs = torch.zeros(inputs.size(0), inputs.size(1), self.n_out)
        #shape: [batch_size, num_atoms, n_out]
        
        for i in range(start_idx, len(self.gcns_1)):
            hidden = F.relu(self.gcns_1[i](inputs))
            hidden = F.dropout(hidden, self.dropout, training=self.training)
            output = self.gcns_2[i](hidden)
            all_outputs += output
            
        return all_outputs
            
        
        


            
        
        
        

