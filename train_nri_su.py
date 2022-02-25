"""Train NRI supervised way"""

from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from data_utils import *
from models_NRI import *

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False, 
                    help="Disables CUDA training.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--no-seed", action="store_true", default=False,
                    help="don't use seed.")
parser.add_argument("--epochs", type=int, default=200, 
                    help="Number of epochs to train.")
parser.add_argument("--batch-size", type=int, default=128,
                    help="Number of samples per batch.")
parser.add_argument("--lr", type=float, default=0.0005,
                    help="Initial learning rate.")
parser.add_argument("--encoder-hidden", type=int, default=256,
                    help="Number of hidden units.")
parser.add_argument("--num-atoms", type=int, default=5,
                    help="Number of atoms.")
parser.add_argument("--encoder", type=str, default="wavenet",
                    help="Type of encoder model.")
parser.add_argument("--no-factor", action="store_true", default=False,
                    help="Disables factor graph model.")
parser.add_argument("--suffix", type=str, default="_static_5",
                    help="Suffix for training data ")
parser.add_argument("--use-motion", action="store_true", default=False,
                    help="use increments")
parser.add_argument("--encoder-dropout", type=float, default=0.3,
                    help="Dropout rate (1-keep probability).")
parser.add_argument("--save-folder", type=str, default="logs/nrisu",
                    help="Where to save the trained model, leave empty to not save anything.")
parser.add_argument("--load-folder", type=str, default='', 
                    help="Where to load the trained model.")
parser.add_argument("--edge-types", type=int, default=2,
                    help="The number of edge types to infer.")
parser.add_argument("--dims", type=int, default=4,
                    help="The number of feature dimensions.")
parser.add_argument("--timesteps", type=int, default=49,
                    help="The number of time steps per sample.")
parser.add_argument("--lr-decay", type=int, default=200,
                    help="After how epochs to decay LR factor of gamma.")
parser.add_argument("--gamma", type=float, default=0.5,
                    help="LR decay factor.")
parser.add_argument("--group-weight", type=float, default=0.5,
                    help="group weight.")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

if not args.no_seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        

log = None
#Save model and meta-data. Always saves in a new folder
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = "{}/exp{}/".format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, "metadata.pkl")
    encoder_file = os.path.join(save_folder, "nri_encoder.pt")
    
    log_file = os.path.join(save_folder, "log.txt")
    log = open(log_file, 'w')
    pickle.dump({"args":args}, open(meta_file, 'wb'))
    
else:
    print("WARNING: No save_folder provided!"+
          "Testing (within this script) will throw an error.")
    

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_spring_sim(
    args.batch_size, args.suffix)


off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.from_numpy(rel_rec)
rel_send = torch.from_numpy(rel_send)

if args.encoder == "mlp":
    encoder = MLPEncoder(args.timesteps*args.dims, args.encoder_hidden, 
                         args.edge_types, args.encoder_dropout, args.factor)

elif args.encoder == "cnn":
    encoder = CNNEncoder(args.dims, args.encoder_hidden, args.edge_types, 
                         args.encoder_dropout, args.factor, use_motion=args.use_motion)

elif args.encoder == "rescnn":
    encoder = ResCausalCNNEncoder(args.dims, args.encoder_hidden, args.edge_types,
                                  do_prob=args.encoder_dropout, factor=args.factor,
                                  use_motion=args.use_motion)
    
elif args.encoder == "wavenet":
    encoder = WavenetEncoder(args.dims, args.encoder_hidden, args.edge_types,
                             do_prob=args.encoder_dropout, factor=args.factor,
                             use_motion=args.use_motion)
    
if args.load_folder:
    encoder_file = os.path.join(args.load_folder, "nri_encoder.pt")
    encoder.load_state_dict(torch.load(encoder_file))
    args.save_folder = False
    

triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)

cross_entropy_weight = torch.tensor([1-args.group_weight, args.group_weight])



if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()
    cross_entropy_weight = cross_entropy_weight.cuda()
    

optimizer = optim.Adam(list(encoder.parameters()),lr=args.lr)
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
    
    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        relations = relations.float()
        relations, relations_masked = relations[:,:,0], relations[:,:,1]
        data = data.float()
        optimizer.zero_grad()
        logits = encoder(data, rel_rec, rel_send)
        #logits shape: [batch_size, n_edges, edge_types]
        
        #Flatten batch dim
        output = logits.view(logits.size(0)*logits.size(1),-1)
        target = relations.view(-1)
        
        loss = F.cross_entropy(output, target.long())
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)
        gp, ngp = edge_precision(logits, relations) #Precision of group and non_group
        gp_train.append(gp)
        ngp_train.append(ngp)
        
        gr,ngr = edge_recall(logits, relations)
        gr_train.append(gr)
        ngr_train.append(ngr)
        
        loss_train.append(loss.item())
        
    
    encoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        relations = relations.float()
        relations, relations_masked = relations[:,:,0], relations[:,:,1]
        
        data = data.float()
        with torch.no_grad():
            logits = encoder(data, rel_rec, rel_send)
            #Shape: [batch_size, n_edges, n_edgetypes]
            
            #Flatten batch dim
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = relations.view(-1)
            loss = F.cross_entropy(output, target.long())
            
            acc = edge_accuracy(logits, relations)
            acc_val.append(acc)
            gp, ngp = edge_precision(logits, relations) #Precision of group and non_group
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
        #torch.save(encoder.state_dict(), encoder_file)
        torch.save(encoder, encoder_file)
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
    encoder.eval()
    #encoder.load_state_dict(torch.load(encoder_file))
    
    
    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        relations = relations.float()
        
        logits = encoder(data, rel_rec, rel_send)
        
        acc = edge_accuracy(logits, relations)
        acc_test.append(acc)
        
        gp, ngp = edge_precision(logits, relations) #Precision of group and non_group
        gp_test.append(gp)
        ngp_test.append(ngp)
        
        gr,ngr = edge_recall(logits, relations)
        gr_test.append(gr)
        ngr_test.append(ngr)
        
    
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
        
    
    
        
            
    
    
            
            
            
            
    
        
    
    

    
    



































