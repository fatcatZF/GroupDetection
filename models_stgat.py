import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class MLP(nn.Module):
    """Two-layer fully-connected ReLU net """
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.dropout_prob = do_prob
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.relu(self.fc2(x))
        return x
    
    



class MotionEmbedding(nn.Module):
    def __init__(self, n_in=4, n_emb=16):
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
        deltaX = X[:,:,1:,:]-X[:,:,:-1,:] #Motion; shape: [batch_size, num_objects, num_timesteps-1, n_in]
        return self.fc(deltaX) #Embeddings of shape: [batch_size, num_objects, num_timesteps-1, n_emb]
    

class GRUCell(nn.Module):
    def __init__(self, n_in=16, n_h=32):
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
    def __init__(self, n_in=16, n_h=32):
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
        x: input at time t; shape: [num_sims, num_atoms, n_in]
        h: hidden at time t-1 ; shape: [num_sims, num_atoms, n_h]
        c: cell at time t-1; shape: [num_sims, num_atoms, n_h]
        """
        i_t = torch.sigmoid(self.l_ii(x)+self.l_hi(h))
        f_t = torch.sigmoid(self.l_if(x)+self.l_hf(h))
        g_t = torch.tanh(self.l_ig(x)+self.l_hg(h))
        o_t = torch.sigmoid(self.l_io(x)+self.l_ho(h))
        c_t = f_t*c + i_t*g_t
        h_t = o_t * torch.tanh(c_t)
        return c_t, h_t
    

class RNN(nn.Module):
    def __init__(self, n_in=16, n_h=32, rnn_type="LSTM", return_sequence=False):
        """
        n_in: input dimensions
        n_h: hidden dimensions
        """
        super(RNN, self).__init__()
        if rnn_type=="LSTM":
            self.rnnCell = LSTMCell(n_in, n_h)
        else: self.rnnCell = GRUCell(n_in, n_h)
        self.n_h = n_h
        self.return_sequence = return_sequence
    def forward(self, X):
        """
        X: input sequence; shape:[batch_size, num_atoms, num_timesteps, n_in]
        output: h; shape: [batch_size, num_atoms, num_timesteps, n_h]
        """
        timesteps = X.size(2) #number of time steps
        h = [] 
        h_i = torch.zeros(X.size(0), X.size(1), self.n_h)
        if isinstance(self.rnnCell, LSTMCell):
            c_i = torch.zeros_like(h_i)
            for i in range(timesteps):
                X_i = X[:,:,i,:]
                c_i, h_i = self.rnnCell(X_i, c_i, h_i)
                h.append(h_i)
        else:
            for i in range(timesteps):
                X_i = X[:,:,i,:]
                h_i = self.rnnCell(X_i, h_i)
                h.append(h_i)
        h = torch.permute(torch.stack(h), (1,2,0,-1)) #shape: [batch_size, num_atoms, num_timesteps, n_h]
        if self.return_sequence:
            return h #return all hidden states
        else: return h[:,:,-1,:] #return the hidden state of the last timestep
        
        
        
        
class GraphAttentionLayer(nn.Module):
    def __init__(self, n_h=32, n_out=16):
        """
        n_h: input dimensions of nodes, corresponding to F
        n_out: output dimensions of nodes, corresponding to F'
        """
        super(GraphAttentionLayer, self).__init__()
        self.fc = nn.Linear(n_h, n_out) #corresponding to W in GAT
        self.a = nn.Linear(2*n_out, 1) #corresponding to a in GAT
    
    def forward(self, h, rel_rec, rel_send):
        """
        h: (hidden) states of all nodes of all batches and timesteps;
           [batch_size, num_atoms, num_timesteps, n_h]
        rel_rec: Edges to receivers; shape: [num_edges, num_atoms]
        rel_send: Edges to senders: shape: [num_edges, num_atoms]
        
        return: h_p: context vector; [batch_size, num_atoms, num_timesteps, n_out]
                A: matrix of edge score
                A_sym: symmetrized version of A
                A_norm: normalized attention score
              
        """
        Wh = self.fc(h) #shape: [batch_size, num_atoms, num_timesteps, n_out]
        rel_rec_e = torch.unsqueeze(rel_rec, 0).expand(Wh.size(0), rel_rec.size(0), rel_rec.size(1))
        receivers = torch.matmul(rel_rec_e, Wh.reshape(Wh.size(0),Wh.size(1),-1)).reshape(Wh.size(0),
                                                                                 rel_rec.size(0),
                                                                                 Wh.size(2),-1)
        #retrive embedding of receivers; shape: [batch_size, num_edges, num_timesteps, n_out]
        rel_send_e = torch.unsqueeze(rel_send, 0).expand(Wh.size(0), rel_send.size(0), rel_send.size(1))
        senders = torch.matmul(rel_send_e, Wh.reshape(Wh.size(0),Wh.size(1),-1)).reshape(Wh.size(0),
                                                                                 rel_send.size(0),
                                                                                 Wh.size(2),-1)
        
        edges = torch.cat([receivers, senders], dim=-1) #shape: [batch_size, num_edges, num_timesteps, 2*n_out(receivers+senders)]
        edges_score = torch.exp(F.leaky_relu(self.a(edges), 0.2)) #shape: [batch_size, num_edges, num_timesteps, 1]
        
        A = edges_score.reshape(edges_score.size(0),edges_score.size(1),edges_score.size(2))
        A = torch.permute(A, (0,-1,1))#shape:[batch_size, num_timesteps, num_edges]
        A = torch.diag_embed(A)
        A = torch.matmul(rel_rec.t(), torch.matmul(A, rel_send)) #shape: [batch_size, num_timesteps, num_atoms(r), num_atoms(s)]
        A_sym = symmetrize(A)
        A_sum = A.sum(-1).unsqueeze(-1)
        A_norm = A/A_sum #Normalized attention score
        h_p = F.relu(torch.matmul(A_norm,torch.permute(Wh, (0,2,1,-1))))
        h_p = torch.permute(h_p, (0, 2, 1,-1))
        
        return h_p, A, A_sym, A_norm
        
        


class STGATEncoder(nn.Module):
    def __init__(self, n_in=4, n_emb_node=16, n_h_node=32, n_inter_out=16,
                 n_motion_out=24, rnn_type="LSTM"):
        """
        n_in: dimension of input features
        n_emb_node: embedding size of node,
        n_h_node: hidden size of node
        rnn_type: LSTM or GRU
        """
        super(STGATEncoder, self).__init__()
        self.motion_embedding = MotionEmbedding(n_in, n_emb_node)
        self.gat = GraphAttentionLayer(n_h_node, n_emb_node) #graph attention layer
        self.motion_rnn = RNN(n_emb_node, n_h_node, rnn_type, return_sequence=True)
        self.interact_rnn = RNN(n_emb_node, n_h_node, rnn_type, return_sequence=False)
        self.interact_out = MLP(n_h_node, 64, n_inter_out)
        self.motion_out = MLP(n_h_node, 64, n_motion_out)       
        
    def forward(self, X, rel_rec, rel_send):
        """
        X: states of batches and timesteps: [batch_size, num_atoms, num_timesteps, num_features]
        rel_rec: [num_edges, num_atoms]
        rel_send: [num_edges, num_atoms]
        """
        em = self.motion_embedding(X) #em:motion embeding:[batch_size, num_atoms, num_timesteps-1, n_emb_node]
        hm = self.motion_rnn(em) #hm: hidden states of motions; [batch_size, num_atoms, num_timesteps-1, n_h_node]
        em_p, A, A_sym, A_norm = self.gat(hm, rel_rec, rel_send) 
        #shape:[batch_size, num_atoms, num_timesteps-1, n_emb_node]
        h_T = self.interact_rnn(em_p) #h: hidden state of interaction at Tobs-1; [batch_size, num_atoms, n_h_node]
        m_out = self.motion_out(hm[:,:,-1,:]) #output of motion hidden; [batch_size, num_atoms, n_motion_out]
        h_T_p = self.interact_out(h_T) #shape: [batch_size, num_atoms, n_inter_out]
        h_T_o = torch.cat([m_out, h_T_p], dim=-1) #shape: [batch_size, num_atoms, n_motion_out+n_inter_out]
        
        return h_T_o, A, A_sym, A_norm
    
    
class STGATDecoder(nn.Module):
    def __init__(self, n_in=4, n_emb_node=16, noise_dim=16, n_h_receive=40, rnn_type = "LSTM"):
        """
        n_in: features of input states
        n_emb_node: dimensions of node embedding
        n_h_node: hidden states of RNN of Decoder
        rnn_type: LSTM or GRU of decoder rnn
        """
        super(STGATDecoder, self).__init__()
        self.noise_dim = noise_dim
        self.n_h_node = noise_dim+n_h_receive
        self.state_embedding = nn.Linear(n_in, n_emb_node)
        self.rnn_type = rnn_type
        if rnn_type == "LSTM":
            self.rnnCell = LSTMCell(n_emb_node, noise_dim+n_h_receive)
        else:
            self.rnnCell = GRUCell(n_emb_node, noise_dim+n_h_receive)
        
        self.out_fc = nn.Linear(noise_dim+n_h_receive, n_in)
        
    def forward(self, X, hd_i, use_steps=None):
        """
        X: sequence of states; [batch_size, num_atoms, num_timesteps, n_h]
        hd: hidden states from encoder; [batch_size, num_atoms, n_h_recieved]
        use_steps: how many steps used as input
        """
        X_i = X[:,:,-1,:] #Final state of sequence; [batch_size, num_atoms, n_in]
        noise = get_noise((hd_i.size(0), hd_i.size(1), self.noise_dim)) #get noise; [batch_size, num_atoms, noise_dim]
        hd_i = torch.cat([hd_i, noise], dim=-1)
        
        if self.rnn_type == "LSTM":
            cd_i = torch.zeros_like(hd_i)
        X_hat = [X_i] #to store predicted states
        num_timesteps = X.size(2)-1
        if use_steps is None:
            use_steps = int(num_timesteps-1)
            
        for i in range(num_timesteps):
            es_i = self.state_embedding(X_i)
            if self.rnn_type=="LSTM":
                cd_i, hd_i = self.rnnCell(es_i, cd_i, hd_i)
            else:
                hd_i = self.rnnCell(es_i, hd_i)
            
            X_i_hat = X_i-self.out_fc(hd_i)
            X_hat.insert(0, X_i_hat)
            
            if i<use_steps:
                X_i = X[:,:,num_timesteps-(i+1),:]
            else:
                X_i = X_i_hat
                
        X_hat = torch.permute(torch.stack(X_hat), (1,2,0,-1))
        return X_hat
            
        
        
        
        
        
        
        
    
        
    
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
