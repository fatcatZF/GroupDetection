import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from utils import *


class GCNLayer(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        """
        args:
            n_in: input features
            n_out: number of output dimensions
        """
        super(GCNLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.weight = Parameter(torch.FloatTensor(n_in, n_out))
        self.skip_weight = Parameter(torch.FloatTensor(n_in, n_out))
        if bias:
            self.bias = Parameter(torch.FloatTensor(n_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.skip_weight.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, A, X=None):
        """
        args:
            X: node features; shape: [batch_size, num_nodes, n_in]
             if X is None, an identity matrix will be used
            A: normalized adjacency matrix (not including self-loops)
               shape: [batch_size, num_nodes, num_nodes]
        """
        num_nodes = A.size(1)
        if X is None:
            X = torch.eye(num_nodes).unsqueeze(0)
            X = X.expand(A.size(0), X.size(1), X.size(2)) #expand to batch size
        
        agg = torch.matmul(torch.matmul(A, X), self.weight) #aggregation of neighbours
        skip = torch.matmul(X, self.skip_weight)
        if self.bias is not None:
          return skip+agg+self.bias
        else:
            return skip+agg
        
        



class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder,self).__init__()
        
    def forward(self, Z):
        """
        args:
            Z: latents variables from GCN_DMoN_Encoder;
              [batch_size, num_nodes, n_latent]
        """
        return torch.sigmoid(torch.matmul(Z, Z.transpose(-1,-2)))  
        #shape:[batch_size, num_nodes, num_nodes]
        
        



class GCNEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_clusters=8):
        """
        args:
            n_in: input features
            n_hid: hidden dimension
            n_out: number of output dimensions
        """
        super(GCNEncoder,self).__init__()
        self.gcn_h = GCNLayer(n_in, n_hid)
        self.bn = nn.BatchNorm1d(n_out)
        self.gcn_o = GCNLayer(n_hid, n_out)
        self.fc_out = nn.Linear(n_out, n_clusters)
        
    def batch_norm(self, X):
        """
        args:
            X:[batch_size, num_nodes, n_hid]
        """
        H = X.view(X.size(0)*X.size(1),-1)
        H = self.bn(H)
        H = H.view(X.size(0),X.size(1),-1)
        return H
        
    def forward(self, A, X):
        """
        args:
            A: normalized adjacency matrix
            X: feature matrix, shape: [batch_size, num_nodes, n_in]
        """
        H = F.selu(self.gcn_h(A, X))
        H = F.selu((self.gcn_o(A, H)))
        H = self.batch_norm(H)
        H = F.softmax(self.fc_out(H),dim=-1)
        
        return H
        
        



            
    