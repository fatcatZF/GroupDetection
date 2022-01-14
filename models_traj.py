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
    causal conv1d layer
    return the sequence with the same length after
    1D causal convolution
    Input: [B, in_channels, L]
    Output: [B, out_channels, L]
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
    
    
    
class ResCausalConvBlock(nn.Module):
    """
    Residual convolutional block, composed sequentially of 2 causal 
    convolutions with Leaky ReLU activation functions, and a parallel 
    residual connection.
    """
    def __init__(self, n_in, n_out, kernel_size, dilation):
        super(ResCausalConvBlock, self).__init__()
        self.conv1 = CausalConv1d(n_in, n_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(n_out, n_out, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.skip_conv = nn.Conv1d(n_in, n_out, 1)
        
    def forward(self, x):
        x_skip = self.skip_conv(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x+x_skip
        return F.leaky_relu(x)
        
        
        
    

     
class CausalCNNEncoder(nn.Module):
    def __init__(self, n_in, c_hidden, c_out, kernel_size,
                 depth, n_out):
        super(CausalCNNEncoder, self).__init__()
        
        res_layers = [] #residual convolutional layers
        for i in range(depth):
            in_channels = n_in if i==0 else c_hidden
            res_layers += [ResCausalConvBlock(in_channels, c_hidden, kernel_size,
                                              dilation=2**i)]
            
        
        self.res_blocks = torch.nn.Sequential(*res_layers)
        self.conv_predict = nn.Conv1d(c_hidden, c_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(c_hidden, 1, kernel_size=1)
        
        self.fc_mu = nn.Linear(c_out, n_out)
        self.fc_sigma = nn.Linear(c_out, n_out)
        
        
    def forward(self, inputs):
        """
        args:
            inputs: [batch_size, num_atoms, num_timesteps, num_features]
            
        return latents of trajectories of atoms
        """
        x = inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2), inputs.size(3))
        #shape: [total_trajectories, num_timesteps, n_in]
        x = x.transpose(-2,-1)
        #shape: [total_trajectories, n_in, num_timesteps]
        x = self.res_blocks(x) #shape: [total_trajectories, c_hidden, num_timesteps]
        pred = self.conv_predict(x)
        
        attention = F.softmax(self.conv_attention(x), dim=-1) #attention over timesteps
        pred_attention = (pred*attention).mean(dim=2) #shape: [total_trajctories, c_out]
        
        pred_attention = pred_attention.view(inputs.size(0), inputs.size(1), -1)
        #shape: [batch_size, num_atoms, c_out]
        
        mu = self.fc_mu(pred_attention)
        sigma = self.fc_sigma(pred_attention)
        
        return mu, sigma
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
            
        
            
    
        
        
        
                
    
        
        