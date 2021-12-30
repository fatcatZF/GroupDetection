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



class CNNTrajEncoder(nn.Module):
    """
    Use 1-dimensional CNN to encode trajectories(sequence)
    """
    def __init__(self, n_in=4, n_emb=16, n_out=8, n_latent = 8,
                 kernel_size=3, num_timesteps=49, model_increment=False):
        """
        variational CNN autoencoder
        n_in: input features of trajectories
        n_emb: embedding dimensions for trajectories
        n_out: final output channels for features
        model_increment: model increments or states
        """
        super(CNNTrajEncoder, self).__init__()
        self.model_increment = model_increment
        self.fc1 = nn.Linear(n_in, n_emb) #embedding layer
        self.conv1 = nn.Conv1d(n_emb, n_out, kernel_size,
                               padding="same", dilation=1)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size,
                               padding="same", dilation=2)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.conv3 = nn.Conv1d(n_out, n_out, kernel_size,
                               padding="same", dilation=4)
        self.bn3 = nn.BatchNorm1d(n_out)
        self.conv4 = nn.Conv1d(n_out, n_out, kernel_size,
                               padding="same", dilation=8)
        self.bn4 = nn.BatchNorm1d(n_out)
        self.conv5 = nn.Conv1d(n_out, n_out, kernel_size,
                               padding="same", dilation=16)
        self.bn5 = nn.BatchNorm1d(n_out)
        
        self.conv_res = nn.Conv1d(n_emb, n_out, kernel_size=1,
                                  padding="same", dilation=1)       
        #residual connection
        
        if model_increment:
            self.fc_mu = nn.Linear((num_timesteps-1)*n_out, n_latent)
            self.fc_sigma = nn.Linear((num_timesteps-1)*n_out, n_latent)
        else:
            self.fc_mu = nn.Linear(num_timesteps*n_out, n_latent)
            self.fc_sigma = nn.Linear(num_timesteps*n_out, n_latent)
            
        self.init_weights()
            
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, X):
        """
        X: [batch_size, num_atoms, num_timesteps, num_features]
        num_sequence = batch_size*num_atoms
        """
        if self.model_increment:
            deltaX = X[:,:,1:,:]-X[:,:,:-1,:] 
            #increments; shape: [batch_size, num_atoms, num_timesteps-1, num_features]
            X_emb = self.fc1(deltaX)
            #embedding shape: [batch_size, num_atoms, num_timesteps-1, n_emb]
        else:
            X_emb = self.fc1(X)
            #embedding shape: [batch_size, num_atoms, num_timesteps, n_emb]
            
        X_permute = X_emb.view(X_emb.size(0)*X_emb.size(1), X_emb.size(2), X_emb.size(3))
        X_permute = X_permute.permute(0, 2, 1)
        #shape:[num_sequences, n_emb, num_timesteps/num_timesteps-1]
        
        X_p = F.relu(self.bn1(self.conv1(X_permute)))
        #shape: [num_sequences, n_out, num_timesteps/num_timesteps-1]
        X_p = F.relu(self.bn2(self.conv2(X_p)))
        X_p = F.relu(self.bn3(self.conv3(X_p)))
        X_p = F.relu(self.bn4(self.conv4(X_p)))
        X_p = self.bn5(self.conv5(X_p))+self.conv_res(X_permute)
        #Skip connection
        X_p = F.relu(X_p)
        #shape: [batch_size*num_atoms, n_out, num_timesteps/num_timesteps-1]
        X_p = X_p.view(X_emb.size(0), X_emb.size(1), X_p.size(1),-1)
        #shape: [batch_size, num_atoms, n_out, num_timesteps/num_timesteps-1]
        X_p = X_p.permute(0,1,-1,2) #[batch_size, num_atoms, num_timesteps, n_out]
        X_p = X_p.reshape(X_p.size(0),X_p.size(1),-1)
        #shape: [batch_size, num_atoms, num_timesteps*n_out]
        
        mu = self.fc_mu(X_p)
        sigma = self.fc_sigma(X_p)
        #shape: [batch_size, num_atoms, n_latent]
        
        return mu, sigma
        
    
class MLPTrajDecoder(nn.Module):
    def __init__(self, n_in=4, n_emb=16, n_latent=8, num_timesteps=49, 
                 model_increment=False):
        super(MLPTrajDecoder, self).__init__()
        self.model_increment = model_increment
        self.num_timesteps = num_timesteps
        if model_increment:
            self.fc1 = nn.Linear(n_latent, (num_timesteps-1)*n_emb)
        else:
            self.fc1 = nn.Linear(n_latent, num_timesteps*n_emb)
            
        self.fc2 = nn.Linear(n_emb, n_in)
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, Z, X):
        """
        Z: latent variables; shape:[batch_size, num_atoms, n_latent]
        X: original sequence: shape: [batch_size, num_atoms, num_timesteps, n_in]
        """
        X_p = F.elu(self.fc1(Z)) #shape: [batch_size, num_atoms, num_timesteps*n_emb/(num_timesteps-1)*n_emb]
        if self.model_increment:
            X_p = X_p.reshape(X_p.size(0), X_p.size(1), self.num_timesteps-1, -1)
        else:
            X_p = X_p.reshape(X_p.size(0), X_p.size(1), self.num_timesteps, -1)
        X_p = self.fc2(X_p) #shape: [batch_size, num_atoms, num_timesteps/num_timesteps-1, n_in]
        
        if self.model_increment:
            X_forward = X[:,:,0,:] #initial states
            X_backward = X[:,:,-1,:] #end states
            X_re_forward = [X_forward]
            X_re_backward = [X_backward]
            for i in range(self.num_timesteps-1):
                delta_i = X_p[:,:,i,:]
                delta_end = X_p[:,:,self.num_timesteps-2-i,:]
                X_forward = X_forward+delta_i
                X_backward = X_backward-delta_end
                X_re_forward.append(X_forward)
                X_re_backward.insert(0, X_backward)
                
            X_re_forward = torch.stack(X_re_forward) #[num_timesteps, batch_size, num_atoms, n_in]
            X_re_backward = torch.stack(X_re_backward)
            X_re_forward = X_re_forward.permute(1,2,0,-1)
            X_re_backward = X_re_backward.permute(1,2,0,-1)
            
            return X_re_forward, X_re_backward
        
        return X_p
            
                
                
            
        
            
    
        
        
        
                
    
        
        