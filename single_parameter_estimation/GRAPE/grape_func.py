# -*- coding: utf-8 -*-
"""
True GRAPE
"""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import time
from Hamil_rl import PhysModel
import Eval
import json
import learning_rate as mylr

class grape_box:
    
    def __init__(self,theta,phi,gamma,time,tau,w0=1):
        self.theta = theta
        self.phi = phi
        self.tools = PhysModel(theta, phi, gamma, w0, time, tau )
        self.tools.Vk = np.zeros((self.tools.Vk).shape)
        #self.tools.Vk[2,:] = -0.51
        self.num = 0
        print('Try GRAPE, tau=%.3f'%tau)
        
    def grape_update(self,tau,mat_sld,ideal):
        dvk = np.copy(self.tools.Vk)
        rho = (self.tools.rho[0,:]+self.tools.rho[1,:])/2
        for t in range(self.tools.interval): # >>> try the first five control when interval//10
            for k in range(3):
                Ma, Mb = self.M123(rho,t+1,k,tau,ideal) # rho at time t+1 determine the gradient of Vk at time t
                dvk[k,t] = ( tau*(mat_sld*mat_sld*Ma).tr() 
                        - 2*(mat_sld*Mb).tr()*tau**2
                        ).real
        
        return dvk
    
    def true_grape(self, dw, steps=50, adjust=[], option='D', ideal=False):
        
        try:
            beta = adjust[0] # from user's define
            print('user defined parameter: \n step ratio(beta) = %.2f%s'%(beta*100,'%'))
        except IndexError:
            beta = 0.01 # learning rate
            print('default parameter: \n  step ratio(beta) = %.2f%s'%(beta*100,'%'))
        print('iteration start, time=0s')
        start_time = time.time()
        qfi_list = []
        counter = 0 # <<< iteration tag
        #self.tools.Vk = np.zeros((self.tools.Vk).shape)
        T, tau = self.tools.time, self.tools.tau
        # pre-calculation
        qfi_old = 0
        rho0, rho1 = self.tools.simple_control(dw, T, option, ideal)
        mat_sld, rho_ave = Eval.tedious_sld(rho0,rho1,dw)
        qfi = Eval.qfisher(rho_ave, mat_sld)/T
        #mat_sld, rho_ave = self.tools.pure_sld(rho0,rho1,dw)
        dVk_old = np.copy(self.tools.Vk)
        weight = mylr.init_weight(self.tools.Vk) # init_weight
        
        # record
        note = open('Log//%s_NO%s Vklog tau_%i_theta_pi%i.txt'%
                    (option,self.num, self.tools.tau*10, 100*self.theta/np.pi),'w')
        header = ['theta', self.theta, 'tau', self.tools.tau, 'time', self.tools.time, 'worker', 0]
        note.write( " ".join(map(lambda x: str(x), header))+'\n') # add headline
        tsne_num = 0
        while counter<=steps:
            
            # record
            note.write( '\n# %f %i\n'%(qfi, tsne_num) )
            _ = [note.write( "  ".join(map(lambda x: str(x), item))+'\n') for item in self.tools.Vk.T]
            note.flush()
            
            dVk = self.grape_update(tau,mat_sld,ideal) # <<< could cause useless calculation, NEED change.
            
            weight = mylr.sep_ada(weight,dVk_old,dVk,delta=0.01) # individual learning rates
            #self.tools.Vk += dVk*weight*beta
            self.tools.Vk += dVk*0.0001
            
            counter += 1 # <<< iteration tag
            rho0, rho1 = self.tools.simple_control(dw, T, option, ideal)
            mat_sld, rho_ave = Eval.tedious_sld(rho0,rho1,dw)
            #qfi = self.tools.qfisher2(rho0,rho1,dw)
            #mat_sld, rho_ave = self.tools.pure_sld(rho0,rho1,dw)
            qfi = Eval.qfisher(rho_ave,mat_sld) / T
            
            delta_qfi = (qfi-qfi_old).real
            if counter%2==0:
                print('finish step %i, runtime=%.2f s'%(counter,time.time()-start_time))
                print('new QFI - old QFI = %.5f'%delta_qfi)
                print('--- ----')
                qfi_old = qfi
            
            dVk_old = dVk
            
            if abs(delta_qfi)<0.0001:
                print('converge')
                counter -= 1
                break
            qfi_list.append(qfi)
            tsne_num += 1


        return qfi, qfi_list, time.time()-start_time, counter
        
    def superevolve(self,Ops,t1,t2,tau,_ideal): 
        # Ops is an operator at time t1
        if t2 <= t1: # No evolving in M123
            return Ops

        t_idx = np.arange(t1,t2,1,dtype=int)
        
        Ops_new = Ops
        for i in t_idx:        
            if _ideal:
                c_ops = []
            else:
                c_ops = [self.tools.Hph[0]]

            medata = qt.mesolve(self.tools.H[i], Ops_new, np.linspace(0,tau,2), c_ops, [])
            Ops_new = medata.states[-1]
            
        return Ops_new # at time slice t2
    
    def superoperate(self,_Hx,_rho):
        
        return _Hx*_rho - _rho*_Hx
        
    def M123(self,rho,j,k,tau,_ideal): 
        tm = self.tools.interval
        Hk = self.tools.smv[k]
        _dxh0 = self.tools.dxh0
        
        Hxrho = self.superoperate(Hk,rho[j])
        
        M1 = 1j*self.superevolve(Hxrho,j,tm,tau,_ideal)
        
        t1_idx = np.arange(1,j+1,1,dtype=int) # start from rho[1] to rho[j]
        M2 = 0
        for i in t1_idx:
            tmp1 = self.superevolve(self.superoperate(_dxh0,rho[i]),i,j,tau,_ideal) # i-1, di, i+1, i+2...
            M2 += self.superevolve(self.superoperate(Hk,tmp1),j,tm,tau,_ideal) #
        M3 = 0
        if j < tm:
            t2_idx = np.arange(j+1,tm+1,1,dtype=int) # start from rho[j+1] to rho[tm]
            for i in t2_idx:
                tmp2 = self.superevolve(Hxrho,j,i,tau,_ideal)
                M3 += self.superevolve(self.superoperate(_dxh0,tmp2),i,tm,tau,_ideal)
        
        return M1, M2+M3
        
if __name__=='__main__':
    tau = 1; T = 10; dw = 0.00001
    _theta = 1.0 * np.pi; _phi = 0
    _gamma = 0.1
    option = 'E'
    
    grape = grape_box(_theta, _phi, _gamma, T, tau, w0=1) 
    ideal = not True
    steps = 500
    qfi, qfi_list, total_time, counter = grape.true_grape(dw, steps, [0.1], option, ideal)
    
    print(qfi/T)
    
    control_rho = (grape.tools.rho[0,:] + grape.tools.rho[1,:])/2
    smv = grape.tools.smv[:]
    smv_expt = np.array(qt.expect(smv, control_rho)) # n_sm, w=1

    tlist = np.arange(0,T+tau,tau)
    plt.plot(tlist,smv_expt[0,:],label=r'$\sigma_x$',alpha=0.5,linewidth=1)
    plt.plot(tlist,smv_expt[1,:],label=r'$\sigma_y$',alpha=0.5,linewidth=1)
    plt.plot(tlist,smv_expt[2,:],label=r'$\sigma_z$',alpha=0.5,linewidth=1)
        
    plt.legend()
    plt.show()
    
    fig, axes = plt.subplots(3, 1)
    for i in range(3):
        axes[i].set_xlim(0, T)
        axes[i].bar(tlist[:-1]+tau/2, grape.tools.Vk[i,:], alpha=0.5, width=tau,)
    plt.show()
    
    fig, ax = plt.subplots(1, 1)    
    ax.plot([i for i in range(counter)],qfi_list,color='lightcoral',label='QFI against GRAPE steps')
    ax.set_xlabel('steps')
    ax.set_ylabel('QFI')
    ax.set_ylim(min(qfi_list)*0.8,)
    ax.annotate('theta=%.2f pi, qfi=%.4f \n total time=%.2f s'%(_theta/np.pi, qfi, total_time), 
                    xy=(counter, qfi_list[-1]),  xycoords='data',
                    xytext=(0.8, 0.8), textcoords='axes fraction',
                    arrowprops=dict(facecolor='coral',width=1),
                    horizontalalignment='right', verticalalignment='top',
                    )
    ax.legend()
    plt.savefig('grape_tau_%i_theta_pi%i.svg'%(tau*10,100*_theta/np.pi))
    plt.show()
    
    
    '''
    theta_list = np.linspace(0,np.pi/2,6) + np.pi/2
    qfi_theta = []
    steps = 500; tmpVk=0;ideal=not True
    grape = grape_box(_theta, _phi, _gamma, T, tau, w0=1) 
    for i, _theta in enumerate(theta_list):
        print('===== =====')
        print(i,_theta)

        qfi, qfi_list, total_time, counter = grape.true_grape(_theta,_phi,_gamma,dw,T,tau,steps,adjust=[0.005],_ideal=ideal)
        
        qfi_theta.append(qfi/T)
        
        qfi_list = [i/T for i in qfi_list]
        fig, ax = plt.subplots(1, 1)    
        ax.plot([i for i in range(counter)],qfi_list,color='lightcoral',label='QFI against GRAPE steps')
        ax.set_xlabel('steps')
        ax.set_ylabel('QFI')
        ax.set_ylim(min(qfi_list)*0.8,)
        ax.annotate('theta=%.2f pi, qfi=%.4f \n total time=%.2f s'%(_theta/np.pi, qfi/T, total_time), 
                    xy=(counter, qfi_list[-1]),  xycoords='data',
                    xytext=(0.8, 0.8), textcoords='axes fraction',
                    arrowprops=dict(facecolor='coral',width=1),
                    horizontalalignment='right', verticalalignment='top',
                    )
        ax.legend()
        plt.savefig('GRAPE_TAU05_theta%.2fpi.png'%(_theta/np.pi))
        plt.show()

        
    plt.scatter(theta_list,qfi_theta,color='darkred',label='GAPER control')
    plt.legend()
    #plt.ylim(0.3,1.1)
    plt.show()

    data = {'QC':'QFI','theta':theta_list.tolist(),'T':T,'QFI':qfi_theta}
    with open('TAU05_QFI GRAPE control against theta %i.json'%total_time,'w') as fl:
        fl.write(json.dumps(data,indent=2))    
        fl.close()
    '''
    '''
    T_list = np.array([1, 3, 5, 8, 10, 15, 20])
    _theta = 0.25 * np.pi
    qfi_T = []
    steps = 500; tmpVk=0;ideal=not True
    for i, t in enumerate(T_list):
        tau = 0.5 # min(t/25,0.2)
        grape = grape_box(_theta, _phi, _gamma, t, tau, w0=1) 
        print('===== =====')
        print(i,t,tau,grape.tools.interval)

        qfi, qfi_list, total_time, counter = grape.true_grape(_theta,_phi,_gamma,dw,t,tau,steps,adjust=[0.01],_ideal=ideal)
        tmpVk = grape.tools.Vk
        
        qfi_T.append(qfi/t)
        
        
    plt.scatter(T_list,qfi_T,color='darkred',label='GAPER control')
    plt.legend()
    plt.ylim(0,)
    plt.show()

    data = {'QC':'QFI','theta':_theta,'T':T_list.tolist(),'QFI':qfi_T}
    with open('QFI GRAPE control against T %i.json'%total_time,'w') as fl:
        fl.write(json.dumps(data,indent=2))    
        fl.close()
    '''
