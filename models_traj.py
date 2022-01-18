"""
Trajectory Representation Learning modules

"""


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
        if self.kernel_size==1:
            return x
        return x[:,:,:-self.padding]




class GatedCausalConv1d(nn.Module):
    """
    Gated Causal Conv1d Layer
    h_(l+1)=tanh(Wg*h_l)*sigmoid(Ws*h_l)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                dilation):
        super(GatedCausalConv1d, self).__init__()
        self.convg = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation) #gate
        self.convs = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation)
        
    def forward(self, x):
        return torch.sigmoid(self.convg(x))*self.convs(x)
            
        

    
    
    
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
        self.skip_conv = CausalConv1d(n_in, n_out, 1, 1)
        
    def forward(self, x):
        x_skip = self.skip_conv(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x+x_skip
        return F.leaky_relu(x)
        
        


class GatedResCausalConvBlock(nn.Module):
    """
    Gated Residual Convolutional block
    """     
    def __init__(self, n_in, n_out, kernel_size, dilation):
        super(GatedResCausalConvBlock, self).__init__()
        self.conv1 = GatedCausalConv1d(n_in, n_out, kernel_size, dilation)
        self.conv2 = GatedCausalConv1d(n_out, n_out, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.skip_conv = CausalConv1d(n_in, n_out, 1, 1)
        
    def forward(self, x):
        x_skip = self.skip_conv(x)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = x+x_skip
        return x
        




     
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
        logvar = self.fc_sigma(pred_attention)
        sigma = torch.exp(0.5*logvar)
        
        return mu, sigma
    




    


class GATLayer(nn.Module):
    """
    Graph Attention Layer
    """
    def __init__(self, n_in, n_emb):
        """
        args:
            n_in: number of node features
            n_emb: embedding dimensions
        """
        super(GATLayer, self).__init__()
        self.fc_spatial_emb = nn.Linear(n_in, n_emb)#to embed spatial relations
        self.fc_node_emb = nn.Linear(n_in, n_emb) #to embed features of nodes
        self.fc_attention = nn.Linear(n_emb, 1)
        
        
    def forward(self, x, rel_rec, rel_send):
        """
        args:
            x: [batch_size, num_atoms, num_timesteps, num_features]
            rel_rec: receiver matrix, [num_edges, num_atoms]
            rel_send: sender matrix, [num_edges, num_atoms]
        """
        #senders and receivers at each moment
        senders = torch.matmul(rel_send, x.view(x.size(0),x.size(1),-1))
        senders = senders.view(senders.size(0), senders.size(1), x.size(2),x.size(3))
        receivers = torch.matmul(rel_rec, x.view(x.size(0),x.size(1),-1))
        receivers = senders.view(receivers.size(0), receivers.size(1), x.size(2),
                      x.size(3))
        #shape: [batch_size, num_edges, num_timesteps, num_features]
        spatial_relations = receivers-senders
        
        nodes_embed = self.fc_node_emb(x)
        #shape: [batch_size, num_atoms, num_timesteps, n_emb]
        spatial_embed = self.fc_spatial_emb(spatial_relations)
        #shape: [batch_size, num_edges, num_timesteps, n_emb]
        
        #reshape nodes and edges, to [batch_size, num_timesteps, num_nodes/edges, n_emb]
        nodes_embed = nodes_embed.permute(0,2,1,-1)
        spatial_embed = spatial_embed.permute(0,2,1,-1)
        
        receivers_embed = torch.matmul(rel_rec, nodes_embed)
        senders_embed = torch.matmul(rel_send, nodes_embed) #shape:[batch_size, num_timesteps, num_edges, n_emb]
        
        #agg = torch.cat([receivers_embed, senders_embed, spatial_embed], dim=-1)
        #shape: [batch_size, num_timesteps, num_edges, 3*n_emb]
        agg = receivers_embed+senders_embed+spatial_embed
        
        attention = F.leaky_relu(self.fc_attention(agg)).squeeze(-1)
        #shape: [batch_size, num_timesteps, num_edges]
        
        left = torch.matmul(rel_rec.t(), torch.diag_embed(attention))
        attention = torch.matmul(left, rel_send) #shape:[batch_size, num_timesteps,num_atoms, num_atoms]
        
        
        attention = F.softmax(attention, dim=-1) #shape:[batch_size, num_timesteps, n_atoms, n_atoms]
        
        
        
        return F.leaky_relu(torch.matmul(attention, nodes_embed))
        
        
 

class EFGAT(nn.Module):
    """Multi-head attention with skip-connection"""
    def __init__(self, n_in, n_emb, n_heads=2):
        super(EFGAT, self).__init__()
        self.fc_skip = nn.Linear(n_in, n_heads*n_emb)
        self.multi_gats = nn.ModuleList([GATLayer(n_in, n_emb) for i in range(n_heads)])
        self.n_heads = n_heads
        
    def forward(self, x, rel_rec, rel_send):
        """
        args: 
            x: [batch_size, num_atoms, num_timesteps, num_features]
            rel_rec: receiver matrix, [num_edges, num_atoms]
            rel_send: sender matrix, [num_edges, num_atoms]
        """
        attention = []
        x_skip = self.fc_skip(x) #shape: [batch_size, num_atoms, num_timesteps, n_heads*n_emb]
        x_skip = x_skip.permute(0,2,1,-1) #shape:[batch_size,num_timesteps,num_atoms, n_heads*n_emb]
        
        
        for i in range(self.n_heads):
            attention.append(self.multi_gats[i](x, rel_rec, rel_send))
            
        attention = torch.cat(attention, dim=-1) #shape:[batch_size, n_timesteps, n_atoms, n_emb]
        
        return (attention+x_skip).permute(0,2,1,-1) #shape:[batch_size,n_atoms,n_timesteps,n_emb]
        
        
        
        


class GraphTCNEncoder(nn.Module):
    """Graph TCN Encoder"""
    def __init__(self, n_in=4, n_emb=16, n_heads=2 ,c_hidden=64, c_out=48, kernel_size=3,
                 depth=5, n_out=32, model_increment=False):
        super(GraphTCNEncoder, self).__init__()
        self.efgat = EFGAT(n_in, n_emb, n_heads)
        
        res_layers = [] #residual convolutional layers
        for i in range(depth):
            in_channels = n_emb*n_heads if i==0 else c_hidden
            res_layers += [GatedResCausalConvBlock(in_channels, c_hidden, kernel_size,
                                              dilation=2**i)]
        self.res_blocks = torch.nn.Sequential(*res_layers)
            
        self.conv_predict = nn.Conv1d(c_hidden, c_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(c_hidden, 1, kernel_size=1)
            
        self.fc_mu = nn.Linear(c_out, n_out)
        self.fc_sigma = nn.Linear(c_out, n_out)
        self.model_increment=model_increment
            
            
    def forward(self, inputs, rel_rec, rel_send):
        """
        args:
            inputs: [batch_size, num_atoms, num_timesteps, num_features]
            
        return: latents of trajectories of atoms
        """
        if self.model_increment:
            inputs = inputs[:,:,1:,:]-inputs[:,:,:-1,:]
        x = self.efgat(inputs, rel_rec, rel_send) #shape: [batch_size, n_atoms, n_timesteps, n_heads*n_emb]
       
        x = x.reshape(x.size(0)*x.size(1),x.size(2),-1) #shape:[total_atoms, n_timesteps, n_heads*n_emb]
        x = x.transpose(-2,-1) #shape:[total_atoms, n_heads*n_emb, n_timesteps]
        #shape: [total_trajectories, n_in, num_timesteps]
        
        x = self.res_blocks(x) #shape: [total_trajectories, c_hidden, num_timesteps]
        pred = self.conv_predict(x)       
        attention = F.softmax(self.conv_attention(x), dim=-1) #attention over timesteps
        pred_attention = (pred*attention).mean(dim=2) #shape: [total_trajctories, c_out]
        
        pred_attention = pred_attention.view(inputs.size(0), inputs.size(1), -1)
        #shape: [batch_size, num_atoms, c_out]
        
        mu = self.fc_mu(pred_attention)
        logvar = self.fc_sigma(pred_attention)
        sigma = torch.exp(0.5*logvar)
        
        return mu, sigma
        
        
        
        
        
        
        
        
    
            
            
            
            
        

    



    

class GRUCell(nn.Module):
    def __init__(self, n_in, n_hid):
        """
        args:
          n_in: dimensions of input features
          n_hid: hidden dimensions
        """
        super(GRUCell, self).__init__()
        self.gru_cell = nn.GRUCell(n_in, n_hid)
        self.n_hid = n_hid
    
    def forward(self, inputs, hidden=None):
        """
        args:
          inputs: features at one timestep
            shape: [batch_size, num_atoms, num_features]
        """
        x = inputs.view(inputs.size(0)*inputs.size(1),-1)
        #shape: [total_atoms, num_features]
        
        
        if hidden is None:
            hidden = torch.zeros(x.size(0),self.n_hid)
            
        x = self.gru_cell(x, hidden) #shape:[total_atoms, n_hid]
    
        return x
    


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
            hc = (hidden, cell)
        h, c = self.lstm_cell(x, hc)
    
        return h, c
    
    


class RNNDecoder(nn.Module):
    def __init__(self, n_latent, n_in, n_emb, n_hid, rnn_type="GRU", 
                 reverse = False):
        """
        args:
          n_latent: dimensions of latent variables
          n_in: dimensions of input
          n_emb: dimensions of embedding 
          n_hid: dimensions of hidden
          reverse: output reverse sequence
        """
        super(RNNDecoder, self).__init__()
        self.fc_latent = nn.Linear(n_latent, n_hid)
        self.fc_embed = nn.Linear(n_in, n_emb)
        self.fc_out = nn.Linear(n_hid, n_in)
        if rnn_type.lower() == "gru":
            self.rnn_cell = GRUCell(n_emb, n_hid)
        else:
            self.rnn_cell = LSTMCell(n_emb, n_hid)
        self.reverse = reverse
      
    def forward(self, latents, inputs):
        """
        args:
          latents: latent variables from encoder;
              shape:[batch_size, num_atoms, n_latent]
          inputs: shape:[batch_size, num_atoms, num_timesteps, n_in]
        """
        num_timesteps = inputs.size(2)
        hidden = torch.tanh(self.fc_latent(latents)) #map latent to initial hidden
        #shape: [batch_size,num_atoms, n_hid]
        hidden = hidden.view(hidden.size(0)*hidden.size(1),-1)
        if isinstance(self.rnn_cell, LSTMCell):
            cell = torch.zeros_like(hidden)
        if self.reverse:
            x = inputs[:,:,-1,:]
        else:
            x = inputs[:,:,0,:] #features at the first timestep
        outputs = [x]
        x_emb = self.fc_embed(x) #shape:[batch_size, num_atoms, n_emb]
        
        if self.reverse:
            for i in range(1, num_timesteps):
                if isinstance(self.rnn_cell, LSTMCell):
                    hidden, cell = self.rnn_cell(x_emb, (hidden, cell))
                else:
                    hidden = self.rnn_cell(x_emb, hidden)
                    #shape: [batch_size*num_atoms, n_hid]
                x_out = self.fc_out(hidden)
                    #shape: [batch_size*num_atoms, num_features]
                x_out = x_out.view(inputs.size(0), inputs.size(1),-1)
                x = x-x_out
                outputs.insert(0, x)
                x_emb = self.fc_embed(x)
                
                
        else:
          for i in range(1, num_timesteps):
                if isinstance(self.rnn_cell, LSTMCell):
                    hidden, cell = self.rnn_cell(x_emb, (hidden, cell))
                else:
                    hidden = self.rnn_cell(x_emb, hidden)
                    #shape: [batch_size*num_atoms, n_hid]
                
                x_out = self.fc_out(hidden)
                #shape: [batch_size*num_atoms, num_features]
                x_out = x_out.view(inputs.size(0), inputs.size(1),-1)
                x = x+x_out #residual connection
                outputs.append(x)
                x_emb = self.fc_embed(x)
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1,2,0,-1)
        return outputs
    
    


        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
            
        
            
    
        
        
        
                
    
        
        