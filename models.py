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
        deltaX = X[:,:,1:,:]-X[:,:,:-1,:] #Motion; shape: [batch_size, num_objects, num_timesteps-1, n_in]
        return self.fc(deltaX) #Embeddings of shape: [batch_size, num_objects, num_timesteps-1, n_emb]
    

class InteractionEmbedding(nn.Module):
    def __init__(self, n_h, n_emb):
        """
        n_h: hidden size of atoms
        n_emb: embedding size of interactions(edges)
        """
        super(InteractionEmbedding, self).__init__()
        self.fc = nn.Linear(2*n_h, n_emb)
        
    def forward(self, h, rel_rec, rel_send):
        """
        h: hidden states of nodes at different timesteps; shape: [batch_size, num_atoms, num_timesteps, n_h]
        rel_rec: Matrix dnoting incomming edges of nodes: shape: [num_edges, num_atoms]
        rel_send: Matrix denoting out edges of nodes; shape: [num_edges_atoms]
        """
        rel_rec_e = torch.unsqueeze(rel_rec, 0)
        rel_send_e = torch.unsqueeze(rel_send, 0)
        rel_rec_e = rel_rec_e.expand(h.size(0),rel_rec.size(0),rel_rec.size(1))
        #shape: [batch_size, num_edges, num_atoms]
        rel_send_e = rel_send_e.expand(h.size(0),rel_send.size(0), rel_send.size(1))
        #flatten the hidden states overtime
        h_re = torch.reshape(h, (h.size(0), h.size(1), -1)) #shape: [batch_size, num_atoms, num_timesteps*n_h]
        receivers = torch.matmul(rel_rec_e, h_re) #shape: [batch_size, num_edgesm num_timesteps*n_h]
        senders = torch.matmul(rel_send_e, h_re)
        receivers = torch.reshape(receivers, (h.size(0), rel_rec.size(0), h.size(2), -1))
        senders = torch.reshape(senders, (h.size(0), rel_send.size(0), h.size(2), -1))
        edges = torch.cat([senders, receivers], dim=-1) #shape: [batch_size, num_edges, num_timesteps, 2*n_h]
        em = self.fc(edges) #Edge(Interaction) embeddings; shape: [batch_size, num_edges, num_timesteps, n_emb]
        return em
        
        
        

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
    def __init__(self, n_in, n_h, rnn_type="LSTM", return_sequence=False):
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






class RNNEncoder(nn.Module):
    def __init__(self, n_in, n_emb_node, n_emb_edge, n_h_node, n_h_edge, rnn_type="LSTM",
                 return_interaction_sequence=False):
        """
        n_in: number of input features
        n_emb_node: node embedding size
        n_emb_edge: edge embedding size
        n_h_node: hidden size of node 
        n_h_edge: hidden size of edge
        """
        super(RNNEncoder, self).__init__()
        self.motion_embedding = MotionEmbedding(n_in, n_emb_node)
        self.interact_embedding = InteractionEmbedding(n_h_node, n_emb_edge)
        self.motion_rnn = RNN(n_emb_node, n_h_node, rnn_type, return_sequence=True)
        self.interact_rnn = RNN(n_emb_edge, n_h_edge, rnn_type, return_sequence=return_interaction_sequence)
        self.fc = nn.Linear(n_h_edge, 1)
        self.return_interaction_sequence = return_interaction_sequence
        
    def forward(self, X, rel_rec, rel_send):
        """
        X: [batch_size, num_atoms, num_timesteps, num_features]
        rel_rec: [num_edges, num_atoms]
        rel_send: [num_edges, num_atoms]
        """
        em = self.motion_embedding(X) #em: motion embedding; [batch_size, num_atoms, num_timesteps-1, num_features]
        hm = self.motion_rnn(em) #hm: hidden states of motions; [batch_size, num_atoms, num_timesteps-1, n_h_node]
        ea = self.interact_embedding(hm, rel_rec, rel_send)
        #ea: interaction embedding; [batch_size, num_edges, num_timesteps-1, n_emb_edge]
        h = self.interact_rnn(ea) 
        #h: hidden states of interactions; if return_interaction_sequence: [batch_size, num_edges, num_timesteps-1, n_h_edge]
        #else: [batch_size, num_edges, n_h_edge]
        
        A =  torch.sigmoid(self.fc(h)) 
        if self.return_interaction_sequence:
            A = torch.permute(A.reshape(A.size(0),A.size(1),A.size(2)), (0,2,1))
        else:
            A = A.reshape(A.size(0), A.size(1)) 
        #A: weights of edges; [batch_size, (num_timesteps-1), num_edges]
        A = torch.diag_embed(A) #size: [batch_size, (num_timesteps-1), num_edges, num_edges]
        #convert to adjacency matrix
        A = torch.matmul(rel_send.t(), torch.matmul(A, rel_rec)) #size: [batch_size, (num_timesteps-1), num_atoms, num_atoms]
        
        
        return symmetrize(A)
        
        




class RNNDecoder(nn.Module):
    def __init__(self, n_in, n_emb_node, n_h_node, rnn_type = "LSTM", 
                 receive_sequence=False):
        """
        n_in: features of input states
        n_emb_node: dimensions of node embedding
        n_h_node: dimensions of node hidden states
        rnn_type: LSTM or GRU
        receive_sequence: whether receive a sequence of interaction matrices
        """
        super(RNNDecoder, self).__init__()
        self.states_embedding = nn.Linear(n_in, n_emb_node)
        self.pooling_embedding = nn.Linear(n_h_node, n_emb_node)
        if rnn_type == "LSTM":
            self.rnnCell = LSTMCell(2*n_emb_node, n_h_node)
        else: self.rnnCell = GRUCell(2*n_emb_node, n_h_node)
        self.out_fc = nn.Linear(n_h_node, n_in)
        self.n_h_node = n_h_node
        self.receive_sequence = receive_sequence
        self.rnn_type = rnn_type
        
    def forward(self, X, A):
        """
        X: sequence of states; [batch_size, num_atoms, num_timesteps, n_in]
        A: Interaction tensor: [batch_size, (num_timesteps-1), num_atoms, num_atoms]
        """
        A = symmetric_normalize(A) #symmetric normalize interaction matrix A
        X_i = X[:,:,0,:] #initial states; [batch_size, num_atoms, n_in]
        es_i = self.states_embedding(X_i) #Initial states embedding; [batch_size, num_atoms, n_emb_node]
        h_i = torch.zeros(X_i.size(0), X_i.size(1), self.n_h_node)
        # h_i: initial hidden states; [batch_size, num_atoms, n_h_node]
        if self.rnn_type == "LSTM":
            c_i = torch.zeros_like(h_i)
        X_hat = [X_i] #predicted states
        num_timesteps = X.size(2)-1
        for i in range(num_timesteps):
            if self.receive_sequence:
                A_i = A[:,i,:,:]
            else: A_i = A
            H_i = torch.matmul(A_i, h_i) #H_i: Social Pooling; [batch_size, num_atoms, n_h_node]
            eh_i = self.pooling_embedding(H_i) #Embedding of Social Pooling; [batch_size, num_atoms, n_emb_node]
            esh_i = torch.cat([es_i, eh_i], dim=-1) 
            #esh_i: concat of embedding of states and social pooling; [batch_size, num_atoms, 2*n_emb_node]
            if self.rnn_type == "LSTM":
                c_i, h_i = self.rnnCell(esh_i, c_i, h_i)
            else:
                h_i = self.rnnCell(esh_i, h_i)
            X_i = X_i+self.out_fc(h_i) #current state; [batch_size, num_atoms, n_in]
            X_hat.append(X_i)
            
        X_hat = torch.permute(torch.stack(X_hat), (1,2,0,-1))
        return X_hat
            
            
        
        
        
        
        
        



















