#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import torch
import torch.multiprocessing as mp
import math, os, json

from common.utils import Counter, file_path, load_net, save_net
from common.J_Envcontinous import Continous_Dephasing_qubit
from common.shared_optim import SharedRMSprop, NewSharedRMSprop, SharedAdam
from model import Net
from train import train
from test_process import extend


os.environ["OMP_NUM_THREADS"] = "1"
parser = argparse.ArgumentParser(description='A3C-PPO')
# Physics model parameters
parser.add_argument('--option', type=str, default='D', help='option: dephasing(D)/emission(E)')
parser.add_argument('--theta', type=float, default=0.25, metavar='θ', help='dephasing angle(ratio): pi/θ')
parser.add_argument('--phi', type=float, default=0, metavar='φ', help='dephasing angle(ratio): pi/φ')
parser.add_argument('--dgamma', type=float, default=0.1, metavar='dγ', help='dephasing rate')
parser.add_argument('--w0', type=int, default=1, help='estimate paramteter')
parser.add_argument('--dw', type=float, default=1e-5, help='estimate paramteter shift for LSD calculation')
parser.add_argument('--maxvk', type=float, default=4.0, help='maximum of |control field(t)|')
parser.add_argument('--time', type=float, default=5, metavar='t', help='time of evolution')
parser.add_argument('--tau', type=float, default=0.1, metavar='τ', help='interval of control field(t)')
# RL parameters
parser.add_argument('--seed', type=int, default=2327, help='Random seed')
parser.add_argument('--share-seed', type=bool, default=False, help='Sharing seed')
parser.add_argument('--num-processes', type=int, default=8, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--max-ep', type=int, default=14000, metavar='STEPS', help='Number of training steps')
parser.add_argument('--update-iter', type=int, metavar='STEPS', help='Max number of partion of forward steps for A3C before update')
parser.add_argument('--max-ep-step', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--gamma', type=float, default=0.9, metavar='γ', help='Discount factor')
parser.add_argument('--lr', type=float, default=1e-4, metavar='η', help='Learning rate')
parser.add_argument('--lr-decay', type=float, default=0.99, metavar='α', help='Learning rate decay factor')
parser.add_argument('--entropy-weight', type=float, default=1e-3, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--clip-gradient-norm', type=bool, default=True, metavar='grad_norm_on_off', help='Gradient L2 normalisation')
parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='grad_norm', help='Gradient L2 normalisation')
parser.add_argument('--td-weight', type=float, default=1.0, metavar='V', help='Entropy regularisation weight')
parser.add_argument('--ppo-N', type=int, default=10, metavar='PPON', help='PPO iteration(update) number')
parser.add_argument('--ppo-epsilon', type=float, default=0.12, metavar='epsilon', help='PPO clip parameter')
parser.add_argument('--ppo-decay', default=False, type=bool, help='Linearly decay ppo epsilon to 0')
parser.add_argument('--load-old', default=False, help='Train from old network')

if __name__ == "__main__":

    args = parser.parse_args()
    args.theta *= math.pi
    args.update_iter = round(args.time/args.tau)
    ## make directory
    if not os.path.exists('Vk'):
        os.mkdir('Vk')
    if not os.path.exists('Net'):
        os.mkdir('Net')
    if not os.path.exists('Log'):
        os.mkdir('Log')
    if not os.path.exists('Process'):
        os.mkdir('Process')

    # init env
    Env_paras = {"maxvk": args.maxvk,
                 "theta": args.theta,
                 "phi": args.phi,
                 "gamma": args.dgamma,
                 "w0": args.w0,
                 "dw": args.dw,
                 "T": args.time,
                 "tau": args.tau,
                 "option": args.option,
                 }
    print('args: ', args)
    # Optimizer # global optimizer
    RMSPparas = {"lr": args.lr,
                 "alpha": args.lr_decay,
                 "eps": 1e-8
                 }

    NRepeat = 1

    for i in range(NRepeat):
        NO = i + 1
        # main process initialize torch seed
        args.seed += 10*NO
        torch.manual_seed(args.seed)
        
        genv = Continous_Dephasing_qubit(**Env_paras)  # global enviorment
        gnet = Net(genv.n_states, genv.n_actions)        # global network
        if args.load_old:
            try:
                gnet = load_net(gnet, args.theta, args.tau, NO)
            except FileNotFoundError:
                print('train from scratch')
                None
        gnet.share_memory()         # share the global parameters in multiprocessing
        #opt = NewSharedRMSprop(gnet.parameters(), **RMSPparas)
        #opt.share_memory()
        Adamparas = {"lr": args.lr}
        opt = SharedAdam(gnet.parameters(), **Adamparas)  ;   

        global_ep, res_queue = Counter(), mp.Queue() # <<< 'i': integer; 'd': double
        res_queue.empty()
        # Parallel Training
        processes = []
        for rank in range(1, args.num_processes+1):
            p = mp.Process(target=train, args=( global_ep, res_queue, args, opt, gnet, genv, NO ),
                            name=rank)
            p.start()
            processes.append(p)

        # record episode reward to plot
        res = []
        qes = []
        while True:
            r = res_queue.get()
            if r is not None:
                res.append(r[0])
                qes.append(r[1])
            else:
                break
        print('max',max(qes))
        # Clean up
        for p in processes:
            p.terminate()
            p.join()

        qes_name = '%s_NO%i-qes_theta_pi%i.json'%(args.option, NO, 100*args.theta/math.pi)
        esdata = {'THETA': args.theta, 'NO': NO, 'QES': qes, 'RES': res}
        with open( file_path('Process/'+qes_name), 'w' ) as fl:  # save vk
            fl.write( json.dumps(esdata) )
            fl.flush()
            fl.close()
            
        save_net(gnet, args.theta, args.tau, NO)




    import matplotlib.pyplot as plt
    plt.style.use('seaborn-bright')
    plt.plot(qes,'r')
    plt.show()
    plt.plot(res,'g')
    plt.show()
    import numpy as np
    qes_8 = np.array(qes).reshape(len(qes)//args.num_processes, args.num_processes)
    qes_1 = np.max(qes_8, axis=1)
    fig, ax = plt.subplots(figsize=(4,3.5))
    ax.plot(qes_1)
    ax.set_xlim(0,1600)
    ax.set_ylim(0,)
    fig.savefig('theta%.2fdgamma%.2fT%idt%.1f.png'%(args.theta/np.pi,args.dgamma,args.time,args.tau))
    plt.show()
    _ = extend(genv, gnet, draw=True)
