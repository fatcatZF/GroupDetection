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

import numpy as np

from utils import *
from data_utils import *
from models_NRI import *

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
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--encoder', type=str, default='rescnn',
                    help='Type of path encoder model (cnn or rescnn).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument("--use-motion", action="store_true", default=False,
                    help="use motion")
parser.add_argument('--suffix', type=str, default='ETH',
                    help="ETH or Hotel")
parser.add_argument('--encoder-dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--save-folder', type=str, default='logs/nriped',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=2,
                    help='The number of input dimensions (position).')
parser.add_argument('--timesteps', type=int, default=15,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=True,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument("--gc-weight", type=float, default=0.,
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
      
if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

#Save model and meta-data
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'nri_encoder.pt')
    decoder_file = os.path.join(save_folder, 'nri_decoder.pt')
    
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
    
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")
    
#Load data
data_folder = os.path.join("data/pedestrian/", args.suffix)

with open(os.path.join(data_folder, "tensors_train.pkl"), 'rb') as f:
    examples_train = pickle.load(f)
with open(os.path.join(data_folder, "labels_train.pkl"), 'rb') as f:
    labels_train = pickle.load(f)
#with open(os.path.join(data_folder, "labels_train_masked.pkl"),'rb') as f:
#    labels_train_masked = pickle.load(f)
with open(os.path.join(data_folder, "tensors_valid.pkl"), 'rb') as f:
    examples_valid = pickle.load(f)
with open(os.path.join(data_folder, "labels_valid.pkl"), 'rb') as f:
    labels_valid = pickle.load(f)
#with open(os.path.join(data_folder, "labels_valid_masked.pkl"), 'rb') as f:
#    labels_valid_masked = pickle.load(f)
with open(os.path.join(data_folder, "tensors_test.pkl"),'rb') as f:
    examples_test = pickle.load(f)
with open(os.path.join(data_folder, "labels_test.pkl"), 'rb') as f:
    labels_test = pickle.load(f)



if args.encoder == 'mlp':
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)

elif args.encoder == 'cnn':
    encoder = CNNEncoder(args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor, use_motion=args.use_motion)

elif args.encoder == "cnnsym":
    encoder = CNNEncoderSym(args.dims, args.encoder_hidden, args.edge_types,
                        do_prob=args.encoder_dropout, factor=args.factor,
                        use_motion=args.use_motion)

    
elif args.encoder=="rescnn":
    encoder = ResCausalCNNEncoder(args.dims, args.encoder_hidden, args.edge_types,
                        do_prob=args.encoder_dropout, factor=args.factor,
                        use_motion=args.use_motion)
    

elif args.encoder == "wavenet":
    encoder = WavenetEncoder(args.dims, args.encoder_hidden, args.edge_types, 
                             kernel_size = args.kernel_size, depth=args.depth,
                             do_prob=args.encoder_dropout, factor=args.factor,
                             use_motion=args.use_motion)
    
elif args.encoder=="wavenetraw":
    encoder = WavenetEncoderRaw(args.dims, args.encoder_hidden, args.edge_types,
                        do_prob=args.encoder_dropout, factor=args.factor,
                        use_motion=False)
    
elif args.encoder=="waveneteuc":
    encoder = WavenetEncoderEuc(args.dims, args.encoder_hidden, args.edge_types,
                        do_prob=args.encoder_dropout, factor=args.factor,
                        use_motion=args.use_motion)

elif args.encoder=="wavenetsym":
    encoder = WavenetEncoderSym(args.dims, args.encoder_hidden, args.edge_types,
                        do_prob=args.encoder_dropout, factor=args.factor,
                        use_motion=args.use_motion)

    
decoder = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'nri_encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'nri_decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))
    args.save_folder = False
    


if args.cuda:
    encoder.cuda()
    decoder.cuda()
    


optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)


def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    acc_train = []
    #co_train = [] #contrastive loss
    gp_train = [] #group precision
    ngp_train = [] #non-group precision
    gr_train = [] #group recall
    ngr_train = [] #non group recall
    
    encoder.train()
    decoder.train()
    
    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices) #to shuffle the training examples
    
    optimizer.zero_grad()
    
    idx_count = 0
    accumulation_steps = min(args.batch_size, len(examples_train))
    
    #loss = 0.
    
    for idx in training_indices:
        example = examples_train[idx]
        label = labels_train[idx]
        #label_masked = labels_train_masked[idx]
        #add batch size
        example = example.unsqueeze(0)
        label = label.unsqueeze(0)
        #label_masked = label_masked.unsqueeze(0)
        num_atoms = example.size(1) #get number of atoms
        rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
        
        if args.cuda:
            example = example.cuda()
            label = label.cuda()
            #label_masked = label_masked.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            
        example = example.float()
        logits = encoder(example, rel_rec, rel_send)
        edges = F.gumbel_softmax(logits, tau=args.temp, hard=args.hard, dim=-1)
        prob = F.softmax(logits, dim=-1)
        #loss_co = args.gc_weight*(torch.mul(prob[:,:,0].float(), label_masked.float()).mean()) #contrasitive loss
        output = decoder(example, edges, rel_rec, rel_send,
                             args.prediction_steps)
        target = example[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)
        loss_kl = kl_categorical_uniform(prob, num_atoms, args.edge_types)
        
        loss = loss_nll+loss_kl
        loss = loss/accumulation_steps
        loss.backward()
        
        idx_count+=1
        
        if idx_count%args.batch_size==0 or idx_count==len(examples_train):
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accumulation_steps = min(args.batch_size, len(examples_train)-idx_count)
        
        acc = edge_accuracy(logits, label)
        acc_train.append(acc)
        gp, ngp = edge_precision(logits, label) #precision of group and non-group
        gp_train.append(gp)
        ngp_train.append(ngp)
        gr, ngr = edge_recall(logits, label) #recall of group and non-group
        gr_train.append(gr)
        ngr_train.append(ngr)
        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        #co_train.append(loss_co.item())
        
        
    #loss.backward()
    #optimizer.step()
    #scheduler.step()
    
    
    nll_val = []
    kl_val = []
    mse_val = []
    #co_val = [] #contrasitive loss
    loss_val = []
    acc_val = []
    gp_val = [] #group precision
    ngp_val = [] #non-group precision
    gr_val = [] #group recall
    ngr_val = [] #non group recall
    
    encoder.eval()
    decoder.eval()
    
    valid_indices = np.arange(len(examples_valid))
    
    
    with torch.no_grad():
        for idx in valid_indices:
            example = examples_valid[idx]
            label = labels_valid[idx]
            #label_masked = labels_valid_masked[idx]
            #add batch size
            example = example.unsqueeze(0)
            label = label.unsqueeze(0)
            #label_masked = label_masked.unsqueeze(0)
            num_atoms = example.size(1) #get number of atoms
            rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
            if args.cuda:
                example = example.cuda()
                label = label.cuda()
                #label_masked = label_masked.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            example = example.float()
            logits = encoder(example, rel_rec, rel_send)
            edges = F.gumbel_softmax(logits, tau=args.temp, hard=True, dim=-1)
            prob = F.softmax(logits, dim=-1)
            #Validation output uses teacher forcing
            #loss_co = args.gc_weight*(torch.mul(prob[:,:,0].float(), label_masked.float()).mean()) #contrasitive loss
            output = decoder(example, edges, rel_rec, rel_send, 1)
            target = example[:, :, 1:, :]
            loss_nll = nll_gaussian(output, target, args.var)
            loss_kl = kl_categorical_uniform(prob, num_atoms, args.edge_types)
            loss_current = loss_nll+loss_kl
            loss_val.append(loss_current.item())
            
            acc = edge_accuracy(logits, label)
            acc_val.append(acc)
            gp, ngp = edge_precision(logits, label) #precision of group and non-group
            gp_val.append(gp)
            ngp_val.append(ngp)
            gr, ngr = edge_recall(logits, label) #recall of group and non-group
            gr_val.append(gr)
            ngr_val.append(ngr)
            mse_val.append(F.mse_loss(output, target).item())
            nll_val.append(loss_nll.item())
            kl_val.append(loss_kl.item())
            #co_val.append(loss_co.item())
            
            
    print('Epoch: {:04d}'.format(epoch+1),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          #'co_train: {:.10f}'.format(np.mean(co_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train)),
          'gr_train: {:.10f}'.format(np.mean(gr_train)),#average group recall
          'ngr_train: {:.10f}'.format(np.mean(ngr_train)), #average non-group recall
          'gp_train: {:.10f}'.format(np.mean(gp_train)), #average group precision
          'ngp_train: {:.10f}'.format(np.mean(ngp_train)),#non-average group precision
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'nll_val: {:.10f}'.format(np.mean(nll_val)),
          'kl_val: {:.10f}'.format(np.mean(kl_val)),
          'mse_val: {:.10f}'.format(np.mean(mse_val)),
          #'co_val: {:.10f}'.format(np.mean(co_val)),
          'acc_val: {:.10f}'.format(np.mean(acc_val)),
          'gr_val: {:.10f}'.format(np.mean(gr_val)),#average group recall
          'ngr_val: {:.10f}'.format(np.mean(ngr_val)), #average non-group recall
          'gp_val: {:.10f}'.format(np.mean(gp_val)), #average group precision
          'ngp_val: {:.10f}'.format(np.mean(ngp_val)),#non-average group precision
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(loss_val) < best_val_loss:
        #torch.save(encoder.state_dict(), encoder_file)
        #torch.save(decoder.state_dict(), decoder_file)
        torch.save(encoder, encoder_file)
        torch.save(decoder, decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch+1),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              #'co_train: {:.10f}'.format(np.mean(co_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'gr_train: {:.10f}'.format(np.mean(gr_train)),#average group recall
              'ngr_train: {:.10f}'.format(np.mean(ngr_train)), #average non-group recall
              'gp_train: {:.10f}'.format(np.mean(gp_train)), #average group precision
              'ngp_train: {:.10f}'.format(np.mean(ngp_train)),#non-average group precision
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              #'co_val: {:.10f}'.format(np.mean(co_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'gr_val: {:.10f}'.format(np.mean(gr_val)),#average group recall
              'ngr_val: {:.10f}'.format(np.mean(ngr_val)), #average non-group recall
              'gp_val: {:.10f}'.format(np.mean(gp_val)), #average group precision
              'ngp_val: {:.10f}'.format(np.mean(ngp_val)),#non-average group precision
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
        
    return np.mean(loss_val)


def test():
    acc_test = []
    gp_test = [] #group precision
    ngp_test = [] #non-group precision
    gr_test = [] #group recall
    ngr_test = [] #non group recall
    nll_test = []
    kl_test = []
    mse_test = []
    encoder = torch.load(encoder_file)
    decoder = torch.load(decoder_file)
    
    encoder.eval()
    decoder.eval()
    
   
    #encoder.load_state_dict(torch.load(encoder_file))
    #decoder.load_state_dict(torch.load(decoder_file))
    
    test_indices = np.arange(len(examples_test))
    
    with torch.no_grad():
        for idx in test_indices:
            example = examples_test[idx]
            label = labels_test[idx]
            #add batch size
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
            edges = F.gumbel_softmax(logits, tau=args.temp, hard=True, dim=-1)
            output = decoder(example, edges, rel_rec, rel_send, 1)
            prob = F.softmax(logits, dim=-1)
            target = example[:, :, 1:, :]
            loss_nll = nll_gaussian(output, target, args.var)
            loss_kl = kl_categorical_uniform(prob, num_atoms, args.edge_types)
            acc = edge_accuracy(logits, label)
            acc_test.append(acc)
            gp, ngp = edge_precision(logits, label) #precision of group and non-group
            gp_test.append(gp)
            ngp_test.append(ngp)
            gr, ngr = edge_recall(logits, label) #recall of group and non-group
            gr_test.append(gr)
            ngr_test.append(ngr)
            mse_test.append(F.mse_loss(output, target).item())
            nll_test.append(loss_nll.item())
            kl_test.append(loss_kl.item())
            
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(nll_test)),
          'kl_test: {:.10f}'.format(np.mean(kl_test)),
          'mse_test: {:.10f}'.format(np.mean(mse_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)),
          'gr_test: {:.10f}'.format(np.mean(gr_test)),#average group recall
          'ngr_test: {:.10f}'.format(np.mean(ngr_test)), #average non-group recall
          'gp_test: {:.10f}'.format(np.mean(gp_test)), #average group precision
          'ngp_test: {:.10f}'.format(np.mean(ngp_test)),#non-average group precision
          )
    
    
    
# Train model
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

test()
    
    
            
        
            
        
        
        
        
        
    
    
    
    
            
            
            
            
            
            
        
    
    
    
    
    
        
        
        
        










    






