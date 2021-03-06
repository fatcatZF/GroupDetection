"""
Using all examples of pedestrain dataset
"""

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

parser.add_argument("--c-hidden", type=int, default=64,
                    help="number of hidden kernels of CNN")
parser.add_argument('--c-out', type=int, default=48,
                    help='out channels of CNN')
parser.add_argument('--n-latent', type=int, default=32,
                    help='latent dimension')
parser.add_argument('--depth', type=int, default=2,
                    help='depth of Residual CNN Blocks')
parser.add_argument("--kernel-size", type=int, default=3, 
                    help="kernel size of CNN")

parser.add_argument("--n-noise", type=int, default=4,
                    help="noise dimension of RNN")
parser.add_argument("--rnn-type", type=str, default="gru",
                    help="rnn cell type in the decoder")
parser.add_argument("--reverse", action="store_true", default=False,
                    help="whether reverse output of rnn decoder.")
parser.add_argument("--teaching-rate", type=float, default=1.,
                    help="Initial Teaching rate.")
parser.add_argument("--teaching-k", type=float, default=1e+3,
                    help="Teaching decay rate.")
parser.add_argument("--min-teaching", type=float, default=1.,
                    help="Minimal Teaching rate")

parser.add_argument('--suffix', type=str, default='ETH',
                    help='Suffix for training data ".')
parser.add_argument('--save-folder', type=str, default='logs/traj',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--dims', type=int, default=2,
                    help='The number of input dimensions.')
parser.add_argument('--timesteps', type=int, default=15,
                    help='The number of time steps per sample.')

parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')


parser.add_argument("--sc-weight", type=float, default=0.2,
                    help = "sparse constraints.")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


print(args)

if not args.no_seed:
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.cuda:
      torch.cuda.manual_seed(args.seed)

initial_teaching_rate = args.teaching_rate


#Load data
data_folder = os.path.join("data/pedestrian", args.suffix)

with open(os.path.join(data_folder, "tensors_all.pkl"), 'rb') as f:
    examples_all = pickle.load(f)
with open(os.path.join(data_folder, "labels_all.pkl"), 'rb') as f:
    labels_all = pickle.load(f)
    

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
    

if args.encoder=="gtcn":
    encoder = GraphTCNEncoder(args.dims, args.n_emb, args.n_heads, args.c_hidden, args.c_out,
                             args.kernel_size, args.depth, args.n_latent, args.model_increment)
    
elif args.encoder=="gcntcn":
    encoder = GCNTCNEncoder(args.dims, args.n_emb, args.c_hidden, args.c_out, args.kernel_size,
                            args.depth, args.n_latent)
    
elif args.encoder=="lstm":
    encoder = LSTMEncoder(args.dims, args.n_emb, args.n_latent)
elif args.encoder=="glstm":
    encoder = GraphLSTMEncoder(args.dims, args.n_emb, args.n_heads, args.n_latent, args.model_increment)
    
elif args.encoder=="gcnlstm":
    encoder = GCNLSTMEncoder(args.dims, args.n_emb, args.n_latent)
    
elif args.encoder=="tcn":
    encoder = TCNEncoder(args.dims, args.n_emb ,args.c_hidden, args.c_out, args.kernel_size,
                         args.depth, args.n_latent)

decoder = RNNDecoder(args.n_latent, args.dims, args.n_emb, args.n_noise,
                     args.rnn_type, args.reverse)


if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'traj_encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'traj_decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))
    args.save_folder = False
    

if args.cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)



def train(epoch, best_train_loss, initial_teaching_rate):
    t = time.time()
    nll_train = []
    #kl_train = []
    mse_train = []
    sc_train = []
    loss_train = []
    
    encoder.train()
    decoder.train()
    
    training_indices = np.arange(len(examples_all))
    np.random.shuffle(training_indices) #to shuffle the training examples
    
    optimizer.zero_grad()
    loss = 0.
    
    for idx in training_indices:
        example = examples_all[idx]
        label = labels_all[idx]
        example = example.unsqueeze(0)
        label = label.unsqueeze(0)
        num_atoms = example.size(1) #get number of atoms
        
        rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
        rel_rec_sl, rel_send_sl = create_edgeNode_relation(num_atoms, self_loops=True)
        
        if args.cuda:
            example = example.cuda()
            label = label.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            rel_rec_sl, rel_send_sl = rel_rec_sl.cuda(), rel_send_sl.cuda()
            
        example = example.float()
        
        teaching_rate = 1.
        
        if isinstance(encoder,GCNTCNEncoder) or isinstance(encoder, GCNLSTMEncoder):
            Z = encoder(example, rel_rec, rel_send)
        else:
        
            Z = encoder(example, rel_rec_sl, rel_send_sl)
            #shape: [batch_size, num_atoms, n_latent]
        
        
        
        loss_sc = args.sc_weight*(torch.norm(Z, p=1, dim=-1).sum())/(Z.size(0)*Z.size(1))
        
        output = decoder(Z, example, teaching_rate)
        
        if args.reverse:
            loss_nll = nll_gaussian(output[:,:,:-1,:], example[:,:,:-1,:], args.var)
            loss_mse = F.mse_loss(output[:,:,:-1,:], example[:,:,:-1,:])
        else:
            loss_nll = nll_gaussian(output[:,:,1:,:], example[:,:,1:,:], args.var)
            loss_mse = F.mse_loss(output[:,:,1:,:], example[:,:,1:,:])
            
        loss_current = loss_nll+loss_sc
        loss += loss_current
        
        mse_train.append(loss_mse.item())
        nll_train.append(loss_nll.item())
        sc_train.append(loss_sc.item())
        
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    loss_train.append(loss.item())
    
    print('Epoch: {:04d}'.format(epoch+1),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          "sc_train: {:.10f}".format(np.mean(sc_train)),
          'time: {:.4f}s'.format(time.time() - t))    
    if args.save_folder and np.mean(loss_train) < best_train_loss and initial_teaching_rate<=args.min_teaching:
        #torch.save(encoder.state_dict(), encoder_file)
        torch.save(encoder, encoder_file)
        #torch.save(decoder.state_dict(), decoder_file)
        torch.save(decoder, decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch+1),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              "sc_train: {:.10f}".format(np.mean(sc_train)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
        
    return np.mean(loss_train), teaching_rate




# Train model
t_total = time.time()
best_train_loss = np.inf
best_epoch = 0

for epoch in range(args.epochs):
    train_loss, initial_teaching_rate = train(epoch, best_train_loss, initial_teaching_rate)
    if train_loss < best_train_loss and initial_teaching_rate<=args.min_teaching:
        best_train_loss = train_loss
        best_epoch = epoch
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch+1))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
    
    
    
    
        
        





