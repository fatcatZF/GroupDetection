import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *



class MotionEmbedding(nn.Module):
    def __init__(self, n_in=4, n_emb=16):
        """
        n_in: number of features
        n_emb: embedding sizes
        """
        super(MotionEmbedding, self).__init__()
        self.fc = nn.Linear(n_in, n_emb)




class CausalConv1d(nn.Module):
    """
    causal conv1d
    return the sequence with the same length after
    1D causal convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                dilation):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = dilation*(kernel_size-1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        """
        shape of x: [total_seq, num_features, num_timesteps]
        """
        x = self.conv(x)
        return x[:,:,:-self.padding]
                
            
        
            
    
        
        
        
                
    
        
        