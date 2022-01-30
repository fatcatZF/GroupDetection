"""
 training graph clustering with fine-tunned nri and 
 trajectory representation learning
"""

from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F


from utils import *
from data_utils import *
from models_NRI import *
from models_traj import *
from models_clustering import *

from sknetwork.topology import get_connected_components


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument("--no-seed", action="store_true", default=False,
                    help="don't use seed")
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')

#Parametres of NRI

parser.add_argument('--nri-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--nri-encoder', type=str, default='rescnn',
                    help='Type of path encoder model (cnn or rescnn).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument("--use-motion", action="store_true", default=False,
                    help="use motion")
parser.add_argument('--suffix', type=str, default='_static_5',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--nri-dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs/pipeline',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--nri-folder', type=str, default='logs/pipeline',
                    help='Where to load the trained nri model.')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')


#Parametres of Traj
parser.add_argument('--traj-folder', type=str, default='logs/pipeline',
                    help='Where to load the trained traj model.')
parser.add_argument("--model-increment", action="store_true",default=False,
                    help="whether model increments.")
parser.add_argument('--traj-emb', type=int, default=16,
                    help='Dimension of embedding')
parser.add_argument('--traj-heads', type=int, default=2,
                    help='Dimension of embedding')
parser.add_argument("--c-hidden", type=int, default=64,
                    help="number of hidden kernels of CNN")
parser.add_argument('--c-out', type=int, default=48,
                    help='out channels of CNN')
parser.add_argument('--traj-latent', type=int, default=32,
                    help='latent dimension')
parser.add_argument('--traj-depth', type=int, default=3,
                    help='depth of Residual CNN Blocks')
parser.add_argument("--kernel-size", type=int, default=5, 
                    help="kernel size of CNN")


#Parametres of GCN
parser.add_argument("--gcn-hid", type=int, default=24,
                    help="GCN hidden size.")
parser.add_argument("--gcn-out", type=int, default=16,
                    help="GCN output size.")
parser.add_argument("--n-clusters", type=int, default=8,
                    help="number of clusters.")






parser.add_argument("--gc-weight", type=float, default=1.,
                    help="Group Contrasitive Weight")





args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)


if not args.no_seed:
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.cuda:
      torch.cuda.manual_seed(args.seed)
      
      
#Save model and meta-data
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    gcn_file = os.path.join(save_folder, 'gcn_encoder.pt')   
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
    
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")
    

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_spring_sim(
    args.batch_size, args.suffix)


rel_rec_sl, rel_send_sl = create_edgeNode_relation(args.num_atoms, self_loops=True)
rel_rec, rel_send = create_edgeNode_relation(args.num_atoms, self_loops=False)






if args.nri_encoder == 'cnn':
    nri_encoder = CNNEncoder(args.dims, args.nri_hidden,
                         args.edge_types,
                         args.nri_dropout, args.factor, use_motion=args.use_motion)
    
elif args.nri_encoder=="rescnn":
    nri_encoder = ResCausalCNNEncoder(args.dims, args.nri_hidden, args.edge_types,
                        do_prob=args.nri_dropout, factor=args.factor,
                        use_motion=args.use_motion)


traj_encoder = GraphTCNEncoder(args.dims, args.traj_emb, args.traj_heads, args.c_hidden, args.c_out,
                         args.kernel_size, args.traj_depth, args.traj_latent, args.model_increment)



nri_encoder_file = os.path.join(os.path.join(args.save_folder, "nri"), 'nri_encoder.pt')
nri_encoder.load_state_dict(torch.load(nri_encoder_file))
traj_encoder_file = os.path.join(os.path.join(args.save_folder, "traj"), 'traj_encoder.pt')
traj_encoder.load_state_dict(torch.load(traj_encoder_file))



gcn_encoder = GCNEncoder(args.traj_latent, args.gcn_hid, args.gcn_out, 
                         args.n_clusters)
#inner_decoder = InnerProductDecoder()#inner product decoder


if args.load_folder:
    gcn_file = os.path.join(args.load_folder, 'gcn_encoder.pt')
    gcn_encoder.load_state_dict(torch.load(gcn_file))
    args.save_folder = False
    
    


if args.cuda:
    nri_encoder = nri_encoder.cuda()
    traj_encoder = traj_encoder.cuda()
    gcn_encoder = gcn_encoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    rel_rec_sl = rel_rec_sl.cuda()
    rel_send_sl = rel_send_sl.cuda()
    
    


optimizer = optim.Adam(list(gcn_encoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)
    





def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []
    adj_loss_train = []
    group_loss_train = [] #group loss
    coll_loss_train = [] #collapse loss
    precision_train = []
    recall_train = []
    F1_train = []
    
    nri_encoder.eval()
    traj_encoder.eval()
    gcn_encoder.train()
    
    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
            #[batch_size, num_nodes, num_timesteps, num_features]
        relations = relations.float()
        relations, relations_masked = relations[:,:,0], relations[:,:,1]
        data = data.float()
        relations_masked = torch.diag_embed(relations_masked)
        relations_masked = torch.matmul(rel_send.t(), 
                                        torch.matmul(relations_masked, rel_rec))
        #shape: [batch_size, num_nodes, num_nodes]
        batch_size = data.size(0)
        num_nodes = relations_masked.size(1)
        
        with torch.no_grad():
            #compute interaction matrix
            edges = nri_encoder(data, rel_rec, rel_send)
            edges = F.softmax(edges, dim=-1)
            A = 1-edges[:,:,0]
            A = torch.matmul(rel_send.t(), torch.matmul(torch.diag_embed(A), rel_rec))
            A = symmetrize(A) #shape: [batch_size, num_atoms, num_atoms]
            A = (A>0.5).float()
            A_norm = normalize_graph(A, add_self_loops=False)
            if args.cuda:
                I = torch.cuda.eye(num_nodes)
            else:
                I = torch.eye(num_nodes)
            I = I.expand(batch_size, num_nodes,num_nodes)
            A_comp = torch.matrix_power(A+I, num_nodes)
            A_comp = (A_comp>0).float()
            A_comp[A_comp==0]=-1
        
            #compute trajectory representation
            mu, sigma = traj_encoder(data, rel_rec_sl, rel_send_sl)
            X = mu+sigma*torch.randn_like(sigma)
            #shape: [batch_size, num_atoms, traj_latent]
        
        optimizer.zero_grad()
        """
        assignments, spectral_loss, collapse_loss = gcn_dmon(A, X)
        loss_co = -args.gc_weight*(torch.matmul(assignments, assignments.transpose(-1,-2))*relations_masked).mean()
        loss = spectral_loss+collapse_loss+loss_co
        """
        Z = gcn_encoder(A_norm, X) 
        
            
        
        
        #prod = inner_decoder(Z)
        #adj_loss = (-A_comp*torch.log(prod)).mean()
        
        adj_loss = (A_comp*torch.cdist(Z,Z, p=1)).mean()
        #group_loss = (-args.gc_weight*(relations_masked*torch.log(prod))).mean()
        group_loss = (args.gc_weight*(relations_masked*torch.cdist(Z,Z, p=1))).mean()
        loss = adj_loss+group_loss
        loss.backward()
        optimizer.step()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                        gamma=args.gamma)
        
        loss_train.append(loss.item())
        adj_loss_train.append(adj_loss.item())
        group_loss_train.append(group_loss.item())
        
        
        
        
    loss_val = []
    adj_loss_val = []
    group_loss_val = [] #group loss
    coll_loss_val = [] #collapse loss
    precision_val = []
    recall_val = []
    F1_val = []
    
    nri_encoder.eval()
    traj_encoder.eval()
    gcn_encoder.eval()
    
    
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = data.float(), relations.float()
        relations, relations_masked = relations[:,:,0], relations[:,:,1]
        relations_masked = torch.diag_embed(relations_masked)
        relations_masked = torch.matmul(rel_send.t(), 
                                        torch.matmul(relations_masked, rel_rec))
        batch_size = data.size(0)
        num_nodes = relations_masked.size(1)
        
        with torch.no_grad():
            #compute interaction matrix
            edges = nri_encoder(data, rel_rec, rel_send)
            edges = F.softmax(edges, dim=-1)
            A = 1-edges[:,:,0]
            A = torch.matmul(rel_send.t(), torch.matmul(torch.diag_embed(A), rel_rec))
            A = symmetrize(A) #shape: [batch_size, num_atoms, num_atoms]
            A = (A>0.5).float()
            A_norm = normalize_graph(A, add_self_loops=False)
            I = torch.eye(num_nodes)
            I = I.expand(batch_size, num_nodes,num_nodes)
            A_comp = torch.matrix_power(A+I, num_nodes)
            A_comp = (A_comp>0).float()
            A_comp[A_comp==0]=-1
        
            #compute trajectory representation
            mu, sigma = traj_encoder(data, rel_rec_sl, rel_send_sl)
            X = mu+sigma*torch.randn_like(sigma)
            #shape: [batch_size, num_atoms, traj_latent]
            Z = gcn_encoder(A_norm, X)
            
            #prod = inner_decoder(Z)
            #adj_loss = (-A_comp*torch.log(prod)).mean()
            
            adj_loss = (A_comp*torch.cdist(Z,Z, p=1)).mean()
            #group_loss = (-args.gc_weight*(relations_masked*torch.log(prod))).mean()
            group_loss = (args.gc_weight*(relations_masked)*torch.cdist(Z,Z, p=1)).mean()
            loss = adj_loss+group_loss
            
            loss_val.append(loss.item())
            adj_loss_val.append(adj_loss.item())
            group_loss_val.append(group_loss.item())
            
            
            
            
            
        print("Epoch: {:04d}".format(epoch+1),
              "loss_train: {:.10f}".format(np.mean(loss_train)),
              "adj_loss_train: {:.10f}".format(np.mean(adj_loss_train)),
              "group_loss_train: {:.10f}".format(np.mean(group_loss_train)),
              "loss_val: {:.10f}".format(np.mean(loss_val)),
              "adj_loss_val: {:.10f}".format(np.mean(adj_loss_val)),
              "group_loss_val: {:.10f}".format(np.mean(group_loss_val)),
              )
        
        
        if args.save_folder and np.mean(loss_val) < best_val_loss:
            torch.save(gcn_encoder.state_dict(), gcn_file)
            print('Best model so far, saving...')
            print("Epoch: {:04d}".format(epoch+1),
                  "loss_train: {:.10f}".format(np.mean(loss_train)),
                  "adj_loss_train: {:.10f}".format(np.mean(adj_loss_train)),
                  "group_loss_train: {:.10f}".format(np.mean(group_loss_train)),
                  "loss_val: {:.10f}".format(np.mean(loss_val)),
                  "adj_loss_val: {:.10f}".format(np.mean(adj_loss_val)),
                  "group_loss_val: {:.10f}".format(np.mean(group_loss_val)),
                  file=log)
            
            log.flush()    
        
            
        return np.mean(loss_val)
    
    
    





#Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch+1))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
    
        
            
            
            
            
    
    
    
    
        
        
        
        
        
        
    
    
    
    





















