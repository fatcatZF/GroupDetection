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
parser.add_argument("--rnn-decoder", type=str, default="gru",
                    help="Type of RNN Decoder.")
parser.add_argument("--use-rnn", action="store_true", default=False,
                    help="whether use RNN decoder.")

parser.add_argument("--decoder-hidden", type=int, default=128,
                    help="hidden of decoder.")
parser.add_argument("--edge-types", type=int, default=2,
                    help="Number of edge types.")

parser.add_argument("--rnn-emb", type=int, default=16,
                    help = "RNN decoder embedding")
parser.add_argument("--rnn-noise", type=int, default=4,
                    help = "Noise dimensions of RNN Decoder.")

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


parser.add_argument('--suffix', type=str, default='ETH',
                    help='Suffix for training data ".')
parser.add_argument('--save-folder', type=str, default='logs/trajsu',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--dims', type=int, default=2,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=15,
                    help='The number of time steps per sample.')

parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument("--do-prob", type=float, default=0.3,
                    help="dropout probability of GNN.")
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')

parser.add_argument("--sc-weight", type=float, default=0.2,
                    help="Sparse Constraint Weight.")

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
    

if args.encoder=="gtcn":
    encoder = GraphTCNEncoder(args.dims, args.n_emb, args.n_heads, args.c_hidden, args.c_out,
                             args.kernel_size, args.depth, args.n_latent, args.model_increment,
                             mode="su")
    
elif args.encoder=="gcntcn":
    encoder = GCNTCNEncoder(args.dims, args.n_emb, args.c_hidden, args.c_out, args.kernel_size,
                            args.depth, args.n_latent,
                            )
    
elif args.encoder=="lstm":
    encoder = LSTMEncoder(args.dims, args.n_emb, args.n_latent)
elif args.encoder=="glstm":
    encoder = GraphLSTMEncoder(args.dims, args.n_emb, args.n_heads, args.n_latent, args.model_increment)
    
elif args.encoder=="gcnlstm":
    encoder = GCNLSTMEncoder(args.dims, args.n_emb, args.n_latent)
    
elif args.encoder=="tcn":
    encoder = TCNEncoder(args.dims, args.n_emb ,args.c_hidden, args.c_out, args.kernel_size,
                         args.depth, args.n_latent, mode="su")  
    
if args.decoder=="gnn":
    decoder = GNNDecoder(args.n_latent, args.decoder_hidden, args.edge_types, do_prob=args.do_prob)
    
elif args.decoder=="concat":
    decoder = ConcatDecoder(args.n_latent, args.decoder_hidden, args.edge_types, do_prob=args.do_prob)
    
else:
    decoder = InnerProdDecoder()
    
    
  
if args.use_rnn:
    rnn_decoder = RNNDecoder(args.n_latent, args.dims, args.rnn_emb, args.rnn_noise,args.rnn_decoder,
                             reverse=False)
  
    

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'traj_encoder.pt')
    #encoder.load_state_dict(torch.load(encoder_file))
    encoder = torch.load(encoder_file)
    decoder_file = os.path.join(args.load_folder, 'traj_decoder.pt')
    #decoder.load_state_dict(torch.load(decoder_file))
    decoder = torch.load(decoder_file)
    if args.use_rnn:
        rnn_decoder_file = os.path.join(args.load_folder, "rnn_decoder.pt")
        rnn_decoder = torch.load(rnn_decoder_file)
    
    
    
    args.save_folder = False
    

if args.cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    if args.use_rnn:
        rnn_decoder = rnn_decoder.cuda()
    
 
#optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
#                       lr=args.lr)

optimizer = optim.SGD(list(encoder.parameters())+list(decoder.parameters()),
                      lr=args.lr)
if args.use_rnn:
    optimizer = optim.SGD(list(encoder.parameters())+list(decoder.parameters())+list(rnn_decoder.parameters()),
                          lr=args.lr)
    

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
    
    if args.use_rnn:
        loss_cross_train = [] #cross entropy loss
        loss_cross_val = [] #val cross entropy loss
        rec_train = [] #reconstruction loss for rnn decoder
        rec_val = [] #reconstruction loss
        sc_train = [] #sparse constraint
        sc_val = []
    
    encoder.train()
    decoder.train()
    if args.use_rnn:
        rnn_decoder.train()
    
    
    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices)
    
    optimizer.zero_grad()
    loss = 0.
    count = 0
    #update_threshold = int(len(examples_train)/4)
    for idx in training_indices:
        example = examples_train[idx]
        label = labels_train[idx]
        
        #add batch size
        example = example.unsqueeze(0)
        #shape: [batch_size, n_atom, n_timesteps, n_dims]
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
        
        if isinstance(encoder,GCNTCNEncoder) or isinstance(encoder, GCNLSTMEncoder):
            Z = encoder(example, rel_rec, rel_send)
        else:
        
            #print("example shape: ", example.size())
            #print("rel_rec_sl shape: ", rel_rec_sl.size())
            #print("rel_send_sl shape: ", rel_send_sl.size())
            Z = encoder(example, rel_rec_sl, rel_send_sl)
            
        logits = decoder(Z, rel_rec, rel_send)
        
        if args.use_rnn:
            example_rec = rnn_decoder(Z, example, teaching_rate=1.) #reconstruction            
            loss_sc = args.sc_weight*(torch.norm(Z, p=1, dim=-1).sum())/(Z.size(0)*Z.size(1))
            loss_rec = nll_gaussian(example_rec[:,:,1:,:], example[:,:,1:,:], args.var)
        
        
        #Flatten batch dim
        output = logits.view(logits.size(0)*logits.size(1),-1)
        target = label.view(-1)
        
        if args.use_rnn:
            loss_cross = F.cross_entropy(output, target.long())
            loss_current = loss_cross+loss_sc+loss_rec
        else:
            loss_current = F.cross_entropy(output, target.long())
        loss = loss+loss_current
        
        count += 1
        
        if (idx+1)%args.batch_size==0 or idx==len(examples_train)-1:
            loss = loss/count
            loss.backward()
            optimizer.step()
            scheduler.step()
            count = 0
            loss = 0.
            optimizer.zero_grad()
        
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
        
        loss_train.append(loss_current.item())
        
        if args.use_rnn:
            loss_cross_train.append(loss_cross)
            rec_train.append(loss_rec)
            sc_train.append(loss_sc)
            
    
    
    
    encoder.eval()
    decoder.eval()
    if args.use_rnn:
        rnn_decoder.eval()
    
    valid_indices = np.arange(len(examples_valid))
    
    with torch.no_grad():
        for idx in valid_indices:
            example = examples_valid[idx]
            label = labels_valid[idx]
            
            #add batch size
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
            
            if isinstance(encoder,GCNTCNEncoder) or isinstance(encoder, GCNLSTMEncoder):
                Z = encoder(example, rel_rec, rel_send)
            else:
            
                Z = encoder(example, rel_rec_sl, rel_send_sl)
                
            logits = decoder(Z, rel_rec, rel_send)
            
            if args.use_rnn:
                example_rec = rnn_decoder(Z, example, teaching_rate=1.) #reconstruction            
                loss_sc = args.sc_weight*(torch.norm(Z, p=1, dim=-1).sum())/(Z.size(0)*Z.size(1))
                loss_rec = nll_gaussian(example_rec[:,:,1:,:], example[:,:,1:,:], args.var)
            
            
            #Flatten batch dim
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = label.view(-1)
            
            #loss_current = F.cross_entropy(output, target.long())
            if args.use_rnn:
                loss_cross = F.cross_entropy(output, target.long())
                loss_current = loss_cross+loss_sc+loss_rec
            else:
                loss_current = F.cross_entropy(output, target.long())
            
            
            acc = edge_accuracy(logits, label)
            acc_val.append(acc)
            gp, ngp = edge_precision(logits, label)
            gp_val.append(gp)
            ngp_val.append(ngp)
            
            gr,ngr = edge_recall(logits, label)
            gr_val.append(gr)
            ngr_val.append(ngr)
            
            loss_val.append(loss_current.item())
            if args.use_rnn:
                loss_cross_val.append(loss_cross)
                rec_val.append(loss_rec)
                sc_val.append(loss_sc)
            
            
            
            if gr==0 or gp==0:
                F1_g = 0
            else:
                F1_g = 2*(gr*gp)/(gr+gp)
            
            #non-group F1
            if ngr==0 or ngp==0:
                F1_ng = 0.
            else:
                F1_ng = 2*(ngr*ngp)/(ngr+ngp)
                
            F1 = 0.5*(F1_g+F1_ng)
                
            F1_val.append(F1)
    
    if args.use_rnn:
             print("Epoch: {:04d}".format(epoch),
                   "loss_train: {:.10f}".format(np.mean(loss_train)),
                   "cross_train: {:.10f}".format(np.mean(loss_cross_train)),
                   "rec_train: {:.10f}".format(np.mean(rec_train)),
                   "sc_train: {:.10f}".format(np.mean(sc_train)),
                   "acc_train: {:.10f}".format(np.mean(acc_train)),
                   "gp_train: {:.10f}".format(np.mean(gp_train)),
                   "ngp_train: {:.10f}".format(np.mean(ngp_train)),
                   "gr_train: {:.10f}".format(np.mean(gr_train)),
                   "ngr_train: {:.10f}".format(np.mean(ngr_train)),
                   "loss_val: {:.10f}".format(np.mean(loss_val)),
                   "cross_val: {:.10f}".format(np.mean(loss_cross_val)),
                   "rec_val: {:.10f}".format(np.mean(rec_val)),
                   "sc_val: {:.10f}".format(np.mean(sc_val)),
                   "acc_val: {:.10f}".format(np.mean(acc_val)),
                   "gp_val: {:.10f}".format(np.mean(gp_val)),
                   "ngp_val: {:.10f}".format(np.mean(ngp_val)),
                   "gr_val: {:.10f}".format(np.mean(gr_val)),
                   "ngr_val: {:.10f}".format(np.mean(ngr_val)),
                   "F1_val: {:.10f}".format(np.mean(F1_val)))   
    
    
    else:
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
        if args.use_rnn:
            torch.save(rnn_decoder, rnn_decoder_file)
        print("Best model so far, saving...")
        
        if args.use_rnn:
            print("Epoch: {:04d}".format(epoch),
                  "loss_train: {:.10f}".format(np.mean(loss_train)),
                  "cross_train: {:.10f}".format(np.mean(loss_cross_train)),
                  "rec_train: {:.10f}".format(np.mean(rec_train)),
                  "sc_train: {:.10f}".format(np.mean(sc_train)),
                  "acc_train: {:.10f}".format(np.mean(acc_train)),
                  "gp_train: {:.10f}".format(np.mean(gp_train)),
                  "ngp_train: {:.10f}".format(np.mean(ngp_train)),
                  "gr_train: {:.10f}".format(np.mean(gr_train)),
                  "ngr_train: {:.10f}".format(np.mean(ngr_train)),
                  "loss_val: {:.10f}".format(np.mean(loss_val)),
                  "cross_val: {:.10f}".format(np.mean(loss_cross_val)),
                  "rec_val: {:.10f}".format(np.mean(rec_val)),
                  "sc_val: {:.10f}".format(np.mean(sc_val)),
                  "acc_val: {:.10f}".format(np.mean(acc_val)),
                  "gp_val: {:.10f}".format(np.mean(gp_val)),
                  "ngp_val: {:.10f}".format(np.mean(ngp_val)),
                  "gr_val: {:.10f}".format(np.mean(gr_val)),
                  "ngr_val: {:.10f}".format(np.mean(ngr_val)),
                  "F1_val: {:.10f}".format(np.mean(F1_val)), file=log)
        else:
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
    if args.use_rnn:
        loss_cross_test = [] #cross entropy loss
        rec_test = [] #reconstruction loss for rnn decoder
        sc_test = [] #sparse constraint
    
    encoder = torch.load(encoder_file)
    decoder = torch.load(decoder_file)
    encoder.eval()
    decoder.eval()
    
    if args.use_rnn:
        rnn_decoder = torch.load(rnn_decoder_file)
        rnn_decoder.eval()
    
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
            rel_rec_sl, rel_send_sl = create_edgeNode_relation(num_atoms, self_loops=True)
            
            if args.cuda:
                example = example.cuda()
                label = label.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                rel_rec_sl, rel_send_sl = rel_rec_sl.cuda(), rel_send_sl.cuda()
                
            example = example.float()
            
            if isinstance(encoder,GCNTCNEncoder) or isinstance(encoder, GCNLSTMEncoder):
                Z = encoder(example, rel_rec, rel_send)
            else:
            
                Z = encoder(example, rel_rec_sl, rel_send_sl)
                
            logits = decoder(Z, rel_rec, rel_send)
            
            if args.use_rnn:
                example_rec = rnn_decoder(Z, example, teaching_rate=1.) #reconstruction            
                loss_sc = args.sc_weight*(torch.norm(Z, p=1, dim=-1).sum())/(Z.size(0)*Z.size(1))
                loss_rec = nll_gaussian(example_rec[:,:,1:,:], example[:,:,1:,:], args.var)
            
            
            #Flatten batch dim
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = label.view(-1)
            
            #loss_current = F.cross_entropy(output, target.long())
            #loss_current = F.cross_entropy(output, target.long())
            if args.use_rnn:
                loss_cross = F.cross_entropy(output, target.long())
                loss_current = loss_cross+loss_sc+loss_rec
            else:
                loss_current = F.cross_entropy(output, target.long())
            
            acc = edge_accuracy(logits, label)
            acc_test.append(acc)
            gp, ngp = edge_precision(logits, label)
            gp_test.append(gp)
            ngp_test.append(ngp)
            
            gr,ngr = edge_recall(logits, label)
            gr_test.append(gr)
            ngr_test.append(ngr)
            
            loss_test.append(loss_current.item())
            if args.use_rnn:
                loss_cross_test.append(loss_cross)
                rec_test.append(loss_rec)
                sc_test.append(loss_sc)
            
            
    
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    if args.use_rnn:
        print('acc_test: {:.10f}'.format(np.mean(acc_test)),
              "gp_test: {:.10f}".format(np.mean(gp_test)),
              "ngp_test: {:.10f}".format(np.mean(ngp_test)),
              "gr_test: {:.10f}".format(np.mean(gr_test)),
              "ngr_test: {:.10f}".format(np.mean(ngr_test)),
              "cross_entropy: {:.10f}".format(np.mean(loss_cross_test)),
              "reconstruction loss: {:.10f}".format(np.mean(rec_test)),
              "sparse constraint loss: {:.10f}".format(np.mean(sc_test))
              )
    else:    
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
            
            
            
            
                
        
    






