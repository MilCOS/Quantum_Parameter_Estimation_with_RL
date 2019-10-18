# test the neural network
# evaluate the generalizability
import numpy as np
import matplotlib.pyplot as plt

from common.utils import v_wrap, load_net, file_path
import common.Eval as Eval
from train import _choose_action as choose_action

def extend(new_env, model, draw=False):
    """test the RL agent in new environment
    new_env: RL environment
    model: trained neural network
    """
    time = new_env.time
    dw = new_env.dw
    tau = new_env.tau
    # Then, we'll generate new Vk
    done = False
    state = new_env.reset()
    while not done:
        action, _, _ = choose_action(model, v_wrap(state), new_env.n_actions)
        new_state, _, done, _ = new_env.step(action)
            
        state = new_state
        # print("State value: ", model.forward(v_wrap(state))[-1])

    Vk = new_env.Vk

    new_env.simple_control(dw, time, option=new_env.option, _ideal=False)

    qfi_time = [0]
    for ii in range(new_env.interval):
        i = ii+1
        cur_time = i*tau
        qfi = Eval.qfisher2(new_env.rho[0, i], new_env.rho[1, i], dw,) / cur_time
        qfi_time.append(qfi)
        
    print('final qfi:', qfi)
    if draw:
        #smv_expt = np.array(qt.expect(new_env.smv, new_env.rho[0,:])) # n_sm, w=1
        #tlist = new_env.tlist
        #plt.plot(tlist,smv_expt[0,:],label=r'$\sigma_x$',alpha=0.9,linewidth=1)
        #plt.plot(tlist,smv_expt[1,:],label=r'$\sigma_y$',alpha=0.9,linewidth=1)
        #plt.plot(tlist,smv_expt[2,:],label=r'$\sigma_z$',alpha=0.9,linewidth=1)
        #plt.show()

        fig, axes = plt.subplots(3, 1)
        for i in range(3):
            axes[i].set_xlim(0, time)
            axes[i].bar(new_env.tlist[:-1]+tau/2, Vk[i, :], alpha=0.5, width=tau,)
        plt.show()
        
        ft = 15
        plt.figure()
        plt.plot(np.arange(0,time+tau,tau), qfi_time, color='r',label=r'new $\theta=%.2f\pi$'%(new_env.theta/np.pi))
        plt.plot(np.arange(0,time+tau,tau), new_env.horizon[:], color='black',label='new baseline')
        plt.xlabel('T',fontsize=ft)
        plt.ylabel('QFI/T',fontsize=ft)
        plt.ylim(0,)
        plt.xlim(0,)
        plt.xticks(fontsize=ft)
        plt.yticks(fontsize=ft)
        plt.legend(fontsize=ft)
        plt.show()
            
    return qfi, qfi_time, Vk


if __name__=='__main__':
    from common.J_Envcontinous import Continous_Dephasing_qubit
    from model import Net
    import json

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

    # w_list

#    w_list = np.arange(0.99, 1-np.pi*2/TIME, -0.01)
#    w_list = np.linspace(0.5, 1.0, 50, endpoint=False)
#    w_list = np.linspace(1.0, 1.5, 51, endpoint=True)
#    w_list = np.arange(1., 1+np.pi*2/TIME, 0.01)
    w_list = [0.5]
#    w_list = [round(w0,2) for w0 in w_list]
    print(w_list, 'Num', len(w_list))
    NN = 100
    #w0 = 1.0
    th_list =  np.arange(0, 0.55, 0.05)

    #for th in th_list:
    for w0 in w_list:
        #if w0 == 1.0: continue
        #THETA = th*np.pi
        qfi_m = []
        qfi_time_m = []
        Vk_m = []; qfi_tmp=0
        print('w0:', w0)
        extend_path = file_path("Net/theta %i tau %.1f W0 %.2f NO %i.json"%(
                THETA*100/np.pi, TAU, w0, NO))
        for i in range(NN):
            Env_paras = {"maxvk": 4,
                 "theta": THETA,
                 "phi": 0.0,
                 "gamma": 0.1,
                 "w0": w0,
                 "dw": 0.00001,
                 "T": TIME,
                 "tau": TAU,
                 "option": OPTION,
                 }
            genv = Continous_Dephasing_qubit(**Env_paras)  # global enviorment
            qfi, qfi_time, Vk = extend(genv, gnet, draw=False) ## generalizability test
            
            qfi_m.append(qfi)
            #qfi_time_m.append(qfi_time)
            #Vk_m.append(Vk.tolist())
            if qfi > qfi_tmp:
                Vk_m = np.copy(Vk)
                qfi_tmp = qfi + 0

        data = {"THETA": THETA,
            "TAU": TAU,
            "W0": w0,
            "QFI": qfi_m,}
            #"QFI T": qfi_time_m,
            #"VK": Vk_m}

        ## store data
        with open(extend_path, 'w') as fl:
            fl.write( json.dumps(data) )
            fl.close()

    


