"""
Train NRI supervised way 
for pedestrian dataset
@author: z fang
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
parser.add_argument("--suffix", type=str, default="ETH",
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
parser.add_argument("--dims", type=int, default=2,
                    help="The number of feature dimensions.")
parser.add_argument("--timesteps", type=int, default=15,
                    help="The number of time steps per sample.")
parser.add_argument("--lr-decay", type=int, default=200,
                    help="After how epochs to decay LR factor of gamma.")
parser.add_argument("--gamma", type=float, default=0.5,
                    help="LR decay factor.")



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
#Save model and meta-data
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
    

#Load data
data_folder = os.path.join("data/pedestrian/all", args.suffix)


with open(os.path.join(data_folder, "tensors_train.pkl"), 'rb') as f:
    examples_train = pickle.load(f)
with open(os.path.join(data_folder, "labels_train.pkl"), 'rb') as f:
    labels_train = pickle.load(f)
with open(os.path.join(data_folder, "tensors_valid.pkl"), 'rb') as f:
    examples_valid = pickle.load(f)
with open(os.path.join(data_folder, "labels_valid.pkl"), 'rb') as f:
    labels_valid = pickle.load(f)
with open(os.path.join(data_folder, "tensors_test.pkl"),'rb') as f:
    examples_test = pickle.load(f)
with open(os.path.join(data_folder, "labels_test.pkl"), 'rb') as f:
    labels_test = pickle.load(f)





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
    
    
if args.cuda:
    encoder.cuda()
    

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
    
    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices)
    
    optimizer.zero_grad()
    loss = 0.
    
    for idx in training_indices:
        example = examples_train[idx]
        label = labels_train[idx]
        #add batch dimension
        example = example.unsqueeze(0)
        label = label.unsqueeze(0)
        num_atoms = example.size(1) #get number of atoms
        rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
        
        if args.cuda:
            example = example.cuda()
            label = label.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            
        example = example.float()
        logits = encoder(example, rel_rec, rel_send)
        #shape: [batch_size, n_edges, n_edgetypes]
        
        output = logits.view(logits.size(0)*logits.size(1),-1)
        target = label.view(-1)
        
        current_loss = F.cross_entropy(output, target.long())
        loss += current_loss
        
        
        #Move tensors back to cpu
        example = example.cpu()
        rel_rec, rel_send = rel_rec.cpu(), rel_send.cpu()
        
        acc = edge_accuracy(logits, label)
        acc_train.append(acc)
        gp, ngp = edge_precision(logits, label)
        gp_train.append(gp)
        ngp_train.append(ngp)
        
        gr,ngr = edge_recall(logits, label)
        gr_train.append(gr)
        ngr_train.append(ngr)
        
        loss_train.append(current_loss.item())
        
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    encoder.eval()
    
    valid_indices = np.arange(len(examples_valid))
    
    with torch.no_grad():
        for idx in valid_indices:
            example = examples_valid[idx]
            label = labels_valid[idx]
            example = example.unsqueeze(0)
            label = label.unsqueeze(0)
            num_atoms = example.size(1)
            rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
            
            if args.cuda:
                example = example.cuda()
                label = label.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            
            example = example.float()
            logits = encoder(example, rel_rec, rel_send)
            
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = label.view(-1)
            loss_current = F.cross_entropy(output, target.long())
            
            #move tensors back to cpu
            example = example.cpu()
            rel_rec, rel_send = rel_rec.cpu(), rel_send.cpu()
            
            acc = edge_accuracy(logits, label)
            acc_val.append(acc)
            gp, ngp = edge_precision(logits, label)
            gp_val.append(gp)
            ngp_val.append(ngp)
            
            gr,ngr = edge_recall(logits, label)
            gr_val.append(gr)
            ngr_val.append(ngr)
            
            loss_val.append(loss_current.item())
            
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
    
    test_indices = np.arange(len(examples_test))
    
    with torch.no_grad():
        for idx in test_indices:
            example = examples_test[idx]
            label = labels_test[idx]
            example = example.unsqueeze(0)
            label = label.unsqueeze(0)
            num_atoms = example.size(1) #get number of atoms
            rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
            if args.cuda:
                example = example.cuda()
                label = label.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            example = example.float()
            logits = encoder(example, rel_rec, rel_send)
            
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = label.view(-1)
            
            acc = edge_accuracy(logits, label)
            acc_test.append(acc)
            gp, ngp = edge_precision(logits, label)
            gp_test.append(gp)
            ngp_test.append(ngp)
            
            gr,ngr = edge_recall(logits, label)
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

            
            
                
            
    
            
            
    
    
    
    
        










