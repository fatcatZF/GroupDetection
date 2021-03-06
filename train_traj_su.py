from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime
import math

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F


from utils import *
from data_utils import *
from models_traj import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument("--no-seed", action="store_true", default=False,
                    help="don't use seed")

parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Initial learning rate.')
parser.add_argument('--n-emb', type=int, default=16,
                    help='Dimension of embedding')
parser.add_argument('--n-heads', type=int, default=2,
                    help='Dimension of embedding')

parser.add_argument("--model-increment", action="store_true", default=False,
                    help="whether model increments in the encoder.")

parser.add_argument("--encoder", type=str, default="gtcn",
                    help="Type of encoder.")
parser.add_argument("--decoder", type=str, default="gnn",
                    help="Type of decoder.")
parser.add_argument("--decoder-hidden", type=int, default=32,
                    help="hidden of decoder.")
parser.add_argument("--edge-types", type=int, default=2,
                    help="Number of edge types.")

parser.add_argument("--c-hidden", type=int, default=64,
                    help="number of hidden kernels of CNN")
parser.add_argument('--c-out', type=int, default=48,
                    help='out channels of CNN')
parser.add_argument('--n-latent', type=int, default=32,
                    help='latent dimension')
parser.add_argument('--depth', type=int, default=3,
                    help='depth of Residual CNN Blocks')
parser.add_argument("--kernel-size", type=int, default=3, 
                    help="kernel size of CNN")


parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--suffix', type=str, default='_static_5',
                    help='Suffix for training data ".')
parser.add_argument('--save-folder', type=str, default='logs/trajsu',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')

parser.add_argument('--lr-decay', type=int, default=100,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument("--do-prob", type=float, default=0.3,
                    help="Dropout probability of decoder.")
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

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
    encoder_file = os.path.join(save_folder, 'traj_encoder.pt')
    decoder_file = os.path.join(save_folder, 'traj_decoder.pt')
    
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



if args.encoder=="gtcn":
    encoder = GraphTCNEncoder(args.dims, args.n_emb, args.n_heads, args.c_hidden, args.c_out,
                             args.kernel_size, args.depth, args.n_latent, args.model_increment,
                             mode="su")
    
elif args.encoder=="gcntcn":
    encoder = GCNTCNEncoder(args.dims, args.n_emb, args.c_hidden, args.c_out, args.kernel_size,
                            args.depth, args.n_latent,
                            mode="su")
    
elif args.encoder=="lstm":
    encoder = LSTMEncoder(args.dims, args.n_emb, args.n_latent)
elif args.encoder=="glstm":
    encoder = GraphLSTMEncoder(args.dims, args.n_emb, args.n_heads, args.n_latent, args.model_increment)
    
elif args.encoder=="gcnlstm":
    encoder = GCNLSTMEncoder(args.dims, args.n_emb, args.n_latent)
    
elif args.encoder=="tcn":
    encoder = TCNEncoder(args.dims, args.n_emb ,args.c_hidden, args.c_out, args.kernel_size,
                         args.depth, args.n_latent)  
    
if args.decoder=="gnn":
    decoder = GNNDecoder(args.n_latent, args.decoder_hidden, args.edge_types, do_prob=args.do_prob)
    
elif args.decoder=="concat":
    decoder = ConcatDecoder(args.n_latent, args.decoder_hidden, args.edge_types, do_prob=args.do_prob)
    
else:
    decoder = InnerProdDecoder()
    

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'traj_encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'traj_decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))
    args.save_folder = False
    
    
triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)




if args.cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    rel_rec_sl = rel_rec_sl.cuda()
    rel_send_sl = rel_send_sl.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()
    
    
#optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=args.lr)

optimizer = optim.SGD(list(encoder.parameters())+list(decoder.parameters()),lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)



def train(epoch, best_val_F1):
    t = time.time()
    loss_train = []
    acc_train = []
    gp_train = []
    ngp_train = []
    gr_train = []
    ngr_train = []
    loss_val = []
    acc_val = []
    gp_val = []
    ngp_val = []
    gr_val = []
    ngr_val = []
    F1_val = []
    
    encoder.train()
    decoder.train()
    
    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = data.float(), relations.float()
        relations, relations_masked = relations[:,:,0], relations[:,:,1]
        optimizer.zero_grad()
        
        if isinstance(encoder,GCNTCNEncoder) or isinstance(encoder, GCNLSTMEncoder):
            Z = encoder(data, rel_rec, rel_send)
        else:
        
            Z = encoder(data, rel_rec_sl, rel_send_sl)
        #shape: [batch_size, n_atoms, n_latent]
        
        logits = decoder(Z, rel_rec, rel_send)
        #shape: [batch_size, n_edges, n_edgetypes]
        
        #Flatten batch dim
        output = logits.view(logits.size(0)*logits.size(1),-1)
        target = relations.view(-1)
        
        loss = F.cross_entropy(output, target.long())
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)
        gp, ngp = edge_precision(logits, relations)
        gp_train.append(gp)
        ngp_train.append(ngp)
        
        gr,ngr = edge_recall(logits, relations)
        gr_train.append(gr)
        ngr_train.append(ngr)
        
        loss_train.append(loss.item())
        
        
    encoder.eval()
    decoder.eval()
    
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = data.float(), relations.float()
        relations, relations_masked = relations[:,:,0], relations[:,:,1]
        
        with torch.no_grad():
            if isinstance(encoder,GCNTCNEncoder) or isinstance(encoder, GCNLSTMEncoder):
                Z = encoder(data, rel_rec, rel_send)
            else:
            
                Z = encoder(data, rel_rec_sl, rel_send_sl)
                
            logits = decoder(Z, rel_rec, rel_send)
            #shape: [batch_size, n_edges, n_edgetypes]
            
            #Flatten batch dim
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = relations.view(-1)
            loss = F.cross_entropy(output, target.long())
            
            acc = edge_accuracy(logits, relations)
            acc_val.append(acc)
            gp, ngp = edge_precision(logits, relations)
            gp_val.append(gp)
            ngp_val.append(ngp)
            
            gr,ngr = edge_recall(logits, relations)
            gr_val.append(gr)
            ngr_val.append(ngr)
            
            loss_val.append(loss.item())
            
            if gr==0 or gp==0:
                F1 = 0
            else:
                F1 = 2*(gr*gp)/(gr+gp)
                
            F1_val.append(F1)
            
    print("Epoch: {:04d}".format(epoch),
          "loss_train: {:.10f}".format(np.mean(loss_train)),
          "acc_train: {:.10f}".format(np.mean(acc_train)),
          "gp_train: {:.10f}".format(np.mean(gp_train)),
          "ngp_train: {:.10f}".format(np.mean(ngp_train)),
          "gr_train: {:.10f}".format(np.mean(gr_train)),
          "ngr_train: {:.10f}".format(np.mean(ngr_train)),
          "loss_val: {:.10f}".format(np.mean(loss_val)),
          "acc_val: {:.10f}".format(np.mean(acc_val)),
          "gp_val: {:.10f}".format(np.mean(gp_val)),
          "ngp_val: {:.10f}".format(np.mean(ngp_val)),
          "gr_val: {:.10f}".format(np.mean(gr_val)),
          "ngr_val: {:.10f}".format(np.mean(ngr_val)),
          "F1_val: {:.10f}".format(np.mean(F1_val)))
    if args.save_folder and np.mean(F1_val) > best_val_F1:
        torch.save(encoder, encoder_file)
        torch.save(decoder, decoder_file)
        print("Best model so far, saving...")
        print("Epoch: {:04d}".format(epoch),
              "loss_train: {:.10f}".format(np.mean(loss_train)),
              "acc_train: {:.10f}".format(np.mean(acc_train)),
              "gp_train: {:.10f}".format(np.mean(gp_train)),
              "ngp_train: {:.10f}".format(np.mean(ngp_train)),
              "gr_train: {:.10f}".format(np.mean(gr_train)),
              "ngr_train: {:.10f}".format(np.mean(ngr_train)),
              "loss_val: {:.10f}".format(np.mean(loss_val)),
              "acc_val: {:.10f}".format(np.mean(acc_val)),
              "gp_val: {:.10f}".format(np.mean(gp_val)),
              "ngp_val: {:.10f}".format(np.mean(ngp_val)),
              "gr_val: {:.10f}".format(np.mean(gr_val)),
              "ngr_val: {:.10f}".format(np.mean(ngr_val)),
              "F1_val: {:.10f}".format(np.mean(F1_val)),
              file=log)
        log.flush()
        
    return np.mean(F1_val)



def test():
    t = time.time()
    loss_test = []
    acc_test = []
    gp_test = []
    ngp_test = []
    gr_test = []
    ngr_test = []
    
    encoder = torch.load(encoder_file)
    decoder = torch.load(decoder_file)
    encoder.eval()
    decoder.eval()
    
    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = data.float(), relations.float()
        
        with torch.no_grad():
            if isinstance(encoder,GCNTCNEncoder) or isinstance(encoder, GCNLSTMEncoder):
                Z = encoder(data, rel_rec, rel_send)
            else:            
                Z = encoder(data, rel_rec_sl, rel_send_sl)
                
            logits = decoder(Z, rel_rec, rel_send)
            #shape: [batch_size, n_edges, n_edgetypes]
            
            #Flatten batch dim
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = relations.view(-1)
            loss = F.cross_entropy(output, target.long())
            
            acc = edge_accuracy(logits, relations)
            acc_test.append(acc)
            gp, ngp = edge_precision(logits, relations)
            gp_test.append(gp)
            ngp_test.append(ngp)
            
            gr,ngr = edge_recall(logits, relations)
            gr_test.append(gr)
            ngr_test.append(ngr)
            
            loss_test.append(loss.item())
            
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('acc_test: {:.10f}'.format(np.mean(acc_test)),
          "gp_test: {:.10f}".format(np.mean(gp_test)),
          "ngp_test: {:.10f}".format(np.mean(ngp_test)),
          "gr_test: {:.10f}".format(np.mean(gr_test)),
          "ngr_test: {:.10f}".format(np.mean(ngr_test))
          )
    
    
    
#Train model
t_total = time.time()
best_val_F1 = -1.
best_epoch = 0

for epoch in range(args.epochs):
    val_F1 = train(epoch, best_val_F1)
    if val_F1 > best_val_F1:
        best_val_F1 = val_F1
        best_epoch = epoch
        
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
test()
    
    
    
                
                
                
        
        
    
    
            
            
            
            
            
                
            
            
        
        
        
        
        
        
    




      





