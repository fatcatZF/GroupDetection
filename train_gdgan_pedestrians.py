"""
Train GD-GAN on pedestrian datasets
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
from models_gdgan import *

from torch.nn.functional import binary_cross_entropy_with_logits as bce

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
parser.add_argument('--lr-g', type=float, default=5e-3,
                    help='Initial generator learning rate.')
parser.add_argument("--lr-d", type=float, default=1e-3,
                    help="Initial discriminator learning rate.")

parser.add_argument("--n-in", type=int, default=2, help="Input dimensions.")
parser.add_argument('--n-emb', type=int, default=16, help='Dimensions of Embedding')
parser.add_argument("--n-hid", type=int, default=32, 
                    help="Dimensions of hidden states.")
parser.add_argument("--n-noise", type=int, default=16, 
                    help="Dimensions of noise.")

parser.add_argument("--suffix", type=str, default="ETH",
                    help="Suffix for training data.")

parser.add_argument("--split", type=str, default="split0",
                    help="Split folder.")

parser.add_argument('--save-folder', type=str, default='logs/gdgan',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--timesteps', type=int, default=15,
                    help='The number of time steps per sample.')

parser.add_argument('--lr-decay', type=int, default=1000,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument("--sc-weight", type=float, default=0.2,
                    help="Sparse Constraint Weight.")


parser.add_argument("--init-g-times", type=int, default=1,
                    help="initial training times of generator in each epoch.")
parser.add_argument("--increase-constant", type=int, default=3,
                    help="increase how many training times of generator.")
parser.add_argument("--increase-g-epochs", type=int, default=10,
                    help="Every n epochs to increase training times of generator.")
parser.add_argument("--max-g-times", type=int, default=22,
                    help="maximal number of training times of generator.")

parser.add_argument("--min-d-loss", type=float, default=0.1,
                    help="minimal loss of discriminator.")
parser.add_argument("--fake-loss-inc", type=float, default=1e-3,
                    help="fake increament of Discriminator loss.")



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
    generator_file = os.path.join(save_folder, 'generator.pt')
    discriminator_file = os.path.join(save_folder, 'discriminator.pt')
    
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
    
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")
    
#Load data
data_folder = os.path.join("data/pedestrian/", args.suffix)



with open(os.path.join(data_folder, "examples_train_unnormalized.pkl"), 'rb') as f:
    examples_train = pickle.load(f)
with open(os.path.join(data_folder, "labels_train.pkl"), 'rb') as f:
    labels_train = pickle.load(f)
with open(os.path.join(data_folder, "examples_valid_unnormalized.pkl"), 'rb') as f:
    examples_valid = pickle.load(f)
with open(os.path.join(data_folder, "labels_valid.pkl"), 'rb') as f:
    labels_valid = pickle.load(f)
with open(os.path.join(data_folder, "examples_test_unnormalized.pkl"),'rb') as f:
    examples_test = pickle.load(f)
with open(os.path.join(data_folder, "labels_test.pkl"), 'rb') as f:
    labels_test = pickle.load(f)
    
examples_train = [torch.from_numpy(example) for example in examples_train]
examples_valid = [torch.from_numpy(example) for example in examples_valid]
examples_test = [torch.from_numpy(example) for example in examples_test]


generator = LSTMGenerator(args.n_in, args.n_emb, args.n_hid, args.n_noise)

discriminator = LSTMDiscriminator(args.n_in, args.n_emb, args.n_hid)


if args.cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    
optimizer_generator = optim.Adam(list(generator.parameters()), lr=args.lr_g)

optimizer_discriminator = optim.Adam(list(discriminator.parameters()), lr=args.lr_d)


scheduler_generator = lr_scheduler.StepLR(optimizer_generator, step_size=args.lr_decay,
                                          gamma=args.gamma)

scheduler_discriminator = lr_scheduler.StepLR(optimizer_discriminator, 
                                              step_size=args.lr_decay,
                                              gamma=args.gamma)



def train_generator():
    loss_train = []
    loss_val = []
    sc_train = []
    sc_val = []
    generator.train()
    discriminator.eval()
    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices)
    
    optimizer_generator.zero_grad()
    idx_count = 0
    accumulation_steps = min(args.batch_size, len(examples_train))
    fake_examples = [] #to store fake examples
    hiddens = [] #to store hidden states
    
    for idx in training_indices:
        example = examples_train[idx]
        #add batch_size
        example = example.unsqueeze(0)
        #shaepe: [0, n_atoms, n_timesteps, n_in]
        n_atoms = example.size(1) #get number of atoms
        n_timesteps = example.size(2) #get timesteps
        T_obs = int(n_timesteps/2)
        rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
        rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)
        
        if args.cuda:
            example = example.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()
        
        example = example.float()
        
        #generate random noise
        noise = torch.randn(1, n_atoms, args.n_noise)
        if args.cuda:
            noise = noise.cuda()
        #generate fake examples and hidden states
        x_fake, hs = generator(example, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
        #x_fake: [1, n_atoms, n_timesteps, n_in]
        #hs: [1, n_atoms, n_h*pred_timesteps]
        
        #compute L1 norm of hidden state hs
        loss_sc = args.sc_weight*(torch.norm(hs, p=1, dim=-1).sum())/(hs.size(0)*hs.size(1))
        sc_train.append(loss_sc.item())
        
        #prediction of discriminator
        out = discriminator(x_fake, rel_rec, rel_send, rel_rec_t, rel_send_t)
        #shape: [1, n_atoms, 1]
        out = out.squeeze(0) #shape: [n_atoms, 1]
        target = torch.ones_like(out)
        #compute classification loss
        loss_class = bce(out, target)
        loss_train.append(loss_class.item())
        
        loss = loss_class+loss_sc
        loss = loss/accumulation_steps
        loss.backward()
        
        idx_count += 1
        
        if idx_count%args.batch_size==0 or idx_count==len(examples_train):
            optimizer_generator.step()
            scheduler_generator.step()
            optimizer_generator.zero_grad()
            accumulation_steps = min(args.batch_size, len(examples_train)-idx_count)
        
        
        fake_examples.append(x_fake)
        hiddens.append(hs)
        
    
    #validation generator    
    generator.eval()
    discriminator.eval()
    valid_indices = np.arange(len(examples_valid))
    with torch.no_grad():
        for idx in valid_indices:
            example = examples_valid[idx]
            #shape: [n_atoms, n_timesteps, n_in]
            example = example.unsqueeze(0)
            #shape: [1, n_atoms, n_timesteps, n_in]
            n_atoms = example.size(1) #get number of atoms
            n_timesteps = example.size(2) #get timesteps
            T_obs = int(n_timesteps/2)
            rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
            rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)
            
            if args.cuda:
                example = example.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()
            
            example = example.float()
            #generate random noise
            noise = torch.randn(1, n_atoms, args.n_noise)
            if args.cuda:
                noise = noise.cuda()
            #generate fake examples and hidden states
            x_fake, hs = generator(example, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
            #x_fake: [1, n_atoms, n_timesteps, n_in]
            #hs: [1, n_atoms, n_h]
            #compute L1 norm of hidden state hs
            loss_sc = args.sc_weight*(torch.norm(hs, p=1, dim=-1).sum())/(hs.size(0)*hs.size(1))
            sc_val.append(loss_sc.item())
            
            #prediction of discriminator
            out = discriminator(x_fake, rel_rec, rel_send, rel_rec_t, rel_send_t)
            #shape: [1, n_atoms, 1]
            out = out.squeeze(0) #shape: [n_atoms, 1]
            target = torch.ones_like(out)
            #compute classification loss
            loss_class = bce(out, target)
            loss_val.append(loss_class.item())
            
            
    return np.mean(loss_train), np.mean(sc_train), np.mean(loss_val), np.mean(sc_val)
            
        






def train_discriminator():
    loss_train = []
    loss_val = []   
    generator.eval()
    discriminator.train()
    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices)
    
    optimizer_discriminator.zero_grad()
    idx_count = 0
    accumulation_steps = min(args.batch_size, len(examples_train))
    
    
    for idx in training_indices:
        example = examples_train[idx]
        #add batch_size
        example = example.unsqueeze(0)
        #shaepe: [1, n_atoms, n_timesteps, n_in]
        n_atoms = example.size(1) #get number of atoms
        n_timesteps = example.size(2) #get timesteps
        T_obs = int(n_timesteps/2)
        rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
        rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)
        
        if args.cuda:
            example = example.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()
        
        example = example.float()
        
        #generate random noise
        noise = torch.randn(1, n_atoms, args.n_noise)
        if args.cuda:
            noise = noise.cuda()
        #generate fake examples and hidden states
        x_fake, hs = generator(example, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
        #x_fake: [1, n_atoms, n_timesteps, n_in]
        #hs: [1, n_atoms, n_h]
        
        #predict fake sequences
        out_fake = discriminator(x_fake, rel_rec, rel_send, rel_rec_t, rel_send_t)
        #shape: [1, n_atoms, 1]       
        out_fake = out_fake.squeeze(0) #shape: [n_atoms, 1]
        target_fake = torch.zeros_like(out_fake)
        #compute fake loss
        loss_fake = bce(out_fake, target_fake)

        #predict true sequences
        out_true = discriminator(example, rel_rec, rel_send, rel_rec_t, rel_send_t)
        #shape: [1, n_atoms, 1]
        out_true = out_true.squeeze(0) #shape: [n_atoms, 1]
        target_true = torch.ones_like(out_true)
        #compute true loss
        loss_true = bce(out_true, target_true)
        
        loss = loss_fake+loss_true
        loss_train.append(loss.item())
        
        loss = loss/accumulation_steps
        loss.backward()
        
        idx_count += 1
        
        if idx_count%args.batch_size==0 or idx_count==len(examples_train):
            optimizer_discriminator.step()
            scheduler_discriminator.step()
            optimizer_discriminator.zero_grad()
            accumulation_steps = min(args.batch_size, len(examples_train)-idx_count)
            
    
    
    #evaluation of discriminator
    generator.eval()
    discriminator.eval()
    valid_indices = np.arange(len(examples_valid))
    
    with torch.no_grad():
        for idx in valid_indices:
            example = examples_valid[idx]
            #shape: [n_atoms, n_timesteps, n_in]
            example = example.unsqueeze(0)
            #shape: [1, n_atoms, n_timesteps, n_in]
            n_atoms = example.size(1) #get number of atoms
            n_timesteps = example.size(2) #get timesteps
            T_obs = int(n_timesteps/2)
            rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
            rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)
            
            if args.cuda:
                example = example.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()
            
            example = example.float()
            #generate random noise
            noise = torch.randn(1, n_atoms, args.n_noise)
            if args.cuda:
                noise = noise.cuda()
            #generate fake examples and hidden states
            x_fake, hs = generator(example, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
            out_fake = discriminator(x_fake, rel_rec, rel_send, rel_rec_t, rel_send_t)
            #shape: [1, n_atoms, 1]       
            out_fake = out_fake.squeeze(0) #shape: [n_atoms, 1]
            target_fake = torch.zeros_like(out_fake)
            #compute fake loss
            loss_fake = bce(out_fake, target_fake)
            
            #predict true sequences
            out_true = discriminator(example, rel_rec, rel_send, rel_rec_t, rel_send_t)
            #shape: [1, n_atoms, 1]
            out_true = out_true.squeeze(0) #shape: [n_atoms, 1]
            target_true = torch.ones_like(out_true)
            #compute true loss
            loss_true = bce(out_true, target_true)
            
            loss = loss_fake+loss_true
            loss_val.append(loss.item())
            
        
    return np.mean(loss_train), np.mean(loss_val)
            
            
            
    
        


def train(epoch, g_times, loss_d_last):
    print("Epoch: {:04d}".format(epoch+1))
    if loss_d_last >= args.min_d_loss:
        loss_train_dis, loss_val_dis = train_discriminator()
        print("loss_train_dis: {:.10f}".format(loss_train_dis),
          "loss_val_dis: {:.10f}".format(loss_val_dis))
    else:
        loss_train_dis = loss_d_last+args.fake_loss_inc
    
    print("Generator Training Times: ", g_times)
    for t in range(g_times):
        loss_train_ge, sc_train_ge, loss_val_ge, sc_val_ge = train_generator()
        print(
          "loss_train_ge: {:.10f}".format(loss_train_ge),
          "sc_train_ge: {:.10f}".format(sc_train_ge),
          "loss_val_ge: {:.10f}".format(loss_val_ge),
          "sc_val_ge: {:.10f}".format(sc_val_ge))
    
    
    if (epoch+1)%10==0:
        torch.save(generator, generator_file)
        torch.save(discriminator, discriminator_file)
        
    return loss_train_dis
        
    
        
    
        
g_times = args.init_g_times   

loss_d_last = np.inf    
        
for epoch in range(args.epochs):
    loss_d_last = train(epoch, g_times, loss_d_last)
    if (epoch+1)%(args.increase_g_epochs)==0:
        g_times = min(g_times+args.increase_constant, args.max_g_times)
    
    
    
print("Optimisation Finished! ")



    


















