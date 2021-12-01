#copied from https://github.com/ethanfetaya/NRI/blob/master/data/generate_dataset.py
# adapted for dynamic interaction graph

from spring_sim import *
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-train", type=int, default=600)
parser.add_argument("--num-valid", type=int, default=200)
parser.add_argument("--num-test", type=int, default=200)
parser.add_argument("--length", type=int, default=5000, help="length of trajectory.")
parser.add_argument("--sample-freq", type=int, default=100, 
                    help="How often to sample the trajectory.")
parser.add_argument("--n-balls", type=int, default=5, help="Number of balls in the simulation.")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--dynamic", action="store_true", default=False, help="whether generate dynamic graphs")
parser.add_argument("--dynamic-rule", type=int, default=0, help="select dynamic rule, 0 represents no dynamic")


args = parser.parse_args()
args.dynamic = bool(args.dynamic and args.dynamic_rule)
print(args)

dynamic_rule = None
suffix = "_static"

if args.dynamic_rule == 1:
    dynamic_rule = dynamic_rule1
    suffix = "_dynamic1"
elif args.dynamic_rule == 2:
    dynamic_rule = dynamic_rule2
    suffix = "_dynamic2"


sim = SpringSim(n_balls=args.n_balls, dynamic=args.dynamic, dynamic_rule=dynamic_rule)

suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)



def generate_dataset(num_sims, length, sample_freq):
    loc_sampled_all = list() #shape: [num_sims, num_sampledTimesteps, num_features, num_atoms]
    vel_sampled_all = list() #shape: [num_sims, num_sampledTimesteps, num_features, num_atoms]
    loc_all = list() #shape: [num_sims, num_timesteps, num_features, num_atoms]
    vel_all = list() #shape: [num_sims, num_timesteps, num_features, num_atoms]
    sampled_indices_all = list() #shape: [num_sims, num_sampledTimesteps]
    if args.dynamic:
        edges_sampled_all = list() #shape: [num_sims, num_sampledTimesteps, num_atoms, num_atoms]
        edges_all = list() #shape: [num_sims, num_timesteps, num_atoms, num_atoms]
    else:
        edges_all = list() #shape: [num_sims, num_atoms, num_atoms]
        
    
    for i in range(num_sims):
        t = time.time()
        # return vectors of one simulation
        if args.dynamic:
            loc_sampled, vel_sampled, loc, vel, edges_sampled,edges, sampled_indices = sim.sample_trajectory(T=length,
        
                                                                                                                 sample_freq=sample_freq)
        else:
            loc_sampled, vel_sampled, loc, vel, edges, sampled_indices = sim.sample_trajectory(T=length,
                                                                                             sample_freq=sample_freq)
            
        if i% 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time()-t))
        
        loc_sampled_all.append(loc_sampled)
        vel_sampled_all.append(vel_sampled)
        loc_all.append(loc)
        vel_all.append(vel)
        sampled_indices_all.append(sampled_indices)
        if args.dynamic:
            edges_sampled_all.append(edges_sampled)
            edges_all.append(edges)
        else:
            edges_all.append(edges)
            
    loc_sampled_all = np.stack(loc_sampled_all)
    vel_sampled_all = np.stack(vel_sampled_all)
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)
    sampled_indices_all = np.stack(sampled_indices_all)
    if args.dynamic:
        edges_sampled_all = np.stack(edges_sampled_all)
        return loc_sampled_all, vel_sampled_all, loc_all, vel_all, edges_sampled_all, edges_all, sampled_indices_all
    return loc_sampled_all, vel_sampled_all, loc_all, vel_all, edges_all, sampled_indices_all



if args.dynamic:
    print("Generating {} training simulations".format(args.num_train))
    loc_sampled_all_train, vel_sampled_all_train, loc_all_train, vel_all_train, edges_sampled_all_train, edges_all_train, sampled_indices_all_train = generate_dataset(args.num_train,
                                                                                                                           args.length,
                                                                                                                           args.sample_freq)
    print("Generating {} validation simulations".format(args.num_valid))
    loc_sampled_all_valid, vel_sampled_all_valid, loc_all_valid, vel_all_valid, edges_sampled_all_valid, edges_all_valid, sampled_indices_all_valid = generate_dataset(args.num_valid,
                                                                                                                           args.length,
                                                                                                                           args.sample_freq)
    print("Generating {} test simulations".format(args.num_test))
    loc_sampled_all_test, vel_sampled_all_test, loc_all_test, vel_all_test, edges_sampled_all_test, edges_all_test, sampled_indices_all_test = generate_dataset(args.num_test,
                                                                                                                           args.length,
                                                                                                                           args.sample_freq)
    

else:
    print("Generating {} training simulations".format(args.num_train))
    loc_sampled_all_train, vel_sampled_all_train, loc_all_train, vel_all_train, edges_all_train, sampled_indices_all_train = generate_dataset(args.num_train,
                                                                                                                           args.length,
                                                                                                                           args.sample_freq)
    print("Generating {} validation simulations".format(args.num_valid))
    loc_sampled_all_valid, vel_sampled_all_valid, loc_all_valid, vel_all_valid,  edges_all_valid, sampled_indices_all_valid = generate_dataset(args.num_valid,
                                                                                                                           args.length,
                                                                                                                           args.sample_freq)
    print("Generating {} test simulations".format(args.num_test))
    loc_sampled_all_test, vel_sampled_all_test, loc_all_test, vel_all_test, edges_all_test, sampled_indices_all_test = generate_dataset(args.num_test,
                                                                                                                           args.length,
                                                                                                                           args.sample_freq)
    
    

np.save('loc_sampled_all_train' + suffix + '.npy', loc_sampled_all_train)
np.save('vel_sampled_all_train' + suffix + '.npy', vel_sampled_all_train)
np.save('loc_all_train' + suffix + '.npy', loc_all_train)
np.save('vel_all_train' + suffix + '.npy', vel_all_train)
np.save("edges_all_train"+suffix+'.npy', edges_all_train)
np.save("sampled_indices_all_train"+suffix+'.npy', sampled_indices_all_train)


np.save('loc_sampled_all_valid' + suffix + '.npy', loc_sampled_all_valid)
np.save('vel_sampled_all_valid' + suffix + '.npy', vel_sampled_all_valid)
np.save('loc_all_valid' + suffix + '.npy', loc_all_valid)
np.save('vel_all_valid' + suffix + '.npy', vel_all_valid)
np.save("edges_all_valid"+suffix+'.npy', edges_all_valid)
np.save("sampled_indices_all_valid"+suffix+'.npy', sampled_indices_all_valid)

np.save('loc_sampled_all_test' + suffix + '.npy', loc_sampled_all_test)
np.save('vel_sampled_all_test' + suffix + '.npy', vel_sampled_all_test)
np.save('loc_all_test' + suffix + '.npy', loc_all_test)
np.save('vel_all_test' + suffix + '.npy', vel_all_test)
np.save("edges_all_test"+suffix+'.npy', edges_all_test)
np.save("sampled_indices_all_test"+suffix+'.npy', sampled_indices_all_test)


if args.dynamic:
    np.save("edges_sampled_all_train"+suffix+'.npy', edges_sampled_all_train)
    np.save("edges_sampled_all_valid"+suffix+'.npy', edges_sampled_all_valid)
    np.save("edges_sampled_all_test"+suffix+'.npy', edges_sampled_all_test)
    



    
    
    
        
    
    
    
    
    
    
    
    
