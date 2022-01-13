from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle 
import os
import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from models_stgat import *
from utils import *
from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False,
                    help="Disables CUDA training.")
parser.add_argument("--seed",type=int,default=44, help="Random seed.")
parser.add_argument("--epochs",type=int,default=100, help="Number of epochs to train.")
parser.add_argument("--save-folder",type=str, default="logs/stgat",
                    help="Where to save the trained model, leave empty to not save anything.")
parser.add_argument("--load-folder",type=str,default='',help="Where to load the trained model if finetunning")
parser.add_argument("--batch-size",type=int, default=128, help="Number of samples per batch.")
parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate.")
parser.add_argument("--lr-decay", type=int, default=200,
                    help="After how many epochs to decay LR by a factor of gamma.")
parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor.")
parser.add_argument("--suffix", type=str, default="_static_5", help="suffix of simulation")
parser.add_argument("--num-atoms",type=int, default=5, help="Number of atoms in simulation.")
parser.add_argument("--dims", type=int, default=4, help="The number of input features")
parser.add_argument("--node-embedding", type=int, default=16, help="Node Embedding Size")
parser.add_argument("--node-hidden", type=int, default=16, help="The hidden size of nodes")
parser.add_argument("--interaction-out", type=int, default=8, help="The output dimension of interaction in encoder")
parser.add_argument("--motion-out", type=int, default=24, help="The output dimension of motions in encoder")
parser.add_argument("--noise-dim", type=int, default=16, help="The hidden size of nodes")
parser.add_argument("--rnn-type", type=str, default="LSTM", help="Use LSTM or GRU")
parser.add_argument("--timesteps", type=int, default=49, help="The number of time steps per sample.")
parser.add_argument("--use-steps", type=int, default=1, 
                    help="Number of steps used as inputs")
parser.add_argument("--var", type=float, default=5e-5, help="Output variance.")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)


n_h_receive = args.motion_out+args.interaction_out #received hidden state for decoder

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda: torch.cuda.manual_seed(args.seed)

if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'stgat_encoder.pt')
    decoder_file = os.path.join(save_folder, 'stgat_decoder.pt')
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!"+" Testing (within this script) will throw an error")
    
train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_spring_sim(
    args.batch_size, args.suffix)


off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.from_numpy(rel_rec)
rel_send = torch.from_numpy(rel_send)

senders = torch.where(rel_send != 0)[1]
receivers = torch.where(rel_rec !=0)[1]

encoder = STGATEncoder(args.dims, args.node_embedding, args.node_hidden, args.interaction_out,
                       args.motion_out, args.rnn_type)
decoder = STGATDecoder(args.dims, args.node_embedding, args.noise_dim, n_h_receive, args.rnn_type)


if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'stgat_encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'stgat_decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))
    args.save_folder = False
    

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)


if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    

def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    encoder.train()
    decoder.train()
    
    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(),relations.cuda()
        data = data.float()
        optimizer.zero_grad()
        h_T_o, A, A_sym, A_norm = encoder(data, rel_rec, rel_send)
        output = decoder(data, h_T_o, )
        use_steps = args.use_steps
        if epoch%10==0:
            print("sampled edge score matrix: ", A_sym[0,-1,:,:])
        loss = nll_gaussian(output[:,:,1:,:], data[:,:,1:,:], args.var)
        loss.backward()
        optimizer.step()
        scheduler.step()
        nll_train.append(loss.item())
        
    
    nll_val = []
    encoder.eval()
    decoder.eval()
    
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(),relations.cuda()
        data = data.float()
        with torch.no_grad():
            h_T_o, A, A_sym, A_norm = encoder(data, rel_rec, rel_send)
            output = decoder(data, h_T_o)
            loss = nll_gaussian(output[:,:,1:,:], data[:,:,1:,:], args.var)
            nll_val.append(loss)
    
    
    print('Epoch: {:04d}'.format(epoch+1),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'nll_val: {:.10f}'.format(np.mean(nll_val)),
          'time: {:.4f}s'.format(time.time() - t))
    
    
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
               'time: {:.4f}s'.format(time.time() - t), file=log
              )
        log.flush()
        
    return np.mean(nll_val)


# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        
print("Optimization Finished")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

    
























