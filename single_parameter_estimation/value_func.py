import numpy as np
import qutip as qt
import common.Hamil_rl as Hamil_rl
import common.Eval as Eval
import json, os

from common.utils import v_wrap, file_path, load_net, save_net
from common.J_Envcontinous import Continous_Dephasing_qubit
from model import Net
import matplotlib.pyplot as plt


def value_func(env, model):
    """used in testing the trained neural network"""
    # Then, we'll generate value along the trajectory
    env.simple_control(genv.dw, env.time, option=env.option, _ideal=False)
    rho0s = env.rho[0] # rho0, rho1
    values = []; mus = []
    for rho in rho0s:
        s = env._translateRHO(rho)
        mu, sigma, v = model.forward(v_wrap(s))
        values += v.data.tolist()
        mus.append(mu.data.numpy())
    
    values = np.array(values)
    mus = np.array(mus).T
    return values, mus

def load_qfi(dirpath, N0, tag, x0):
    filelist = os.listdir(dirpath)
    for file in filelist:
        filepath = os.path.join(dirpath, file)
        if "NO %i"%NO not in file: continue
        with  open(filepath, 'r') as fl:
            qda = json.load(fl)
            fl.close()
        if tag == 'w0':
            x = round(qda['W0'], 6)
        elif tag == 'th':
            continue
        else:
            raise IndexError
        
        if x not in x0:
            continue
        qfi_NN = np.array(qda['QFI'])
        Vk_NN = np.array(qda["VK"])
        qfi_mid = np.argsort(qfi_NN)[-1]
        Vk = Vk_NN[qfi_mid]
        print(qfi_NN[qfi_mid],Vk.shape)
        
    return Vk

def load_grape(dirpath, theta, tau):
    filelist = os.listdir(dirpath)
    for file in filelist:
        filepath = os.path.join(dirpath, file)
        with  open(filepath, 'r') as fl:
            qda = json.load(fl)
            fl.close()
        if 'T10' in file: continue
            
        if qda["theta"] != theta:
            continue
        if round(qda["tlist"][1]-qda["tlist"][0],3) != tau:
            continue

        print(file)
        Vk = np.array(qda["VK"])
        Vk = Vk.reshape(Vk.size//3, 3).T
        try:
            print(qda["qfi"], Vk.shape)
        except KeyError:
            print(qda["QFI"], Vk.shape)
        
    return Vk

if __name__=='__main__':

    OPTION = "D"
    THETA = np.pi * 0.25
    TIME = 5; TAU = 0.1
    NO = 1
    Env_paras = {"maxvk": 4,
                 "theta": THETA,
                 "phi": 0.0,
                 "gamma": 0.1,
                 "w0": 1,
                 "dw": 0.00001,
                 "T": TIME,
                 "tau": TAU,
                 "option": OPTION,
                 }
    genv = Continous_Dephasing_qubit(**Env_paras)  # global enviorment
    gnet = Net(genv.n_states, genv.n_actions)        # global network
    # load
    gnet = load_net(gnet, THETA, TAU, NO,)# ppo=False)
    # Vk
    w_list = [1.0]
    filepath = file_path("Net/theta%itau0.1_extend_of_w0"%(THETA*100/np.pi))
    #Vk = load_qfi(filepath, NO, "w0", w_list)
    genv.Vk = np.zeros_like(genv.Vk)
    value_rl, mu_rl = value_func(genv, gnet)
    
    # GRAPE
    filepath = file_path("Net/grape_sample")
    Vk = load_grape(filepath, THETA, TAU)
    genv.Vk = Vk
    value_gp, mu_gp = value_func(genv, gnet)
    
    f1, ax = plt.subplots(figsize=(4,4))
    
    ax.plot(np.arange(value_gp.size), value_gp, 's', color='red', alpha=0.5)
    ax.plot(np.arange(value_rl.size), value_rl, '-', lw=1.0, color='blue')
    
    f2, axes = plt.subplots(3,1, figsize=(4,4))
    for i in range(mu_gp[:,1].size):
        axes[i].plot(mu_gp[i], '-', lw=1.0, color='orange'); 
        axes[i].plot(mu_rl[i], '-',lw=1.0,color='blue')
