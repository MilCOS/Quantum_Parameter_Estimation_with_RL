# -*- coding: utf-8 -*-
"""
Environment: two level dephasing

@author: Junning LI

"""

import numpy as np
import qutip as qt
import common.Hamil_rl as Hamil_rl
import common.Eval as Eval
import common.PM_aid as PM_aid# For error message


class Continous_Dephasing_qubit(Hamil_rl.PhysModel, object):
    def __init__(self, maxvk, theta, phi, gamma, w0, dw, T, tau, option):
        print('--- ---')
        print('env init')
        super(Continous_Dephasing_qubit, self).__init__(theta, phi, gamma, w0, T, tau)
        self.maxvk = maxvk
        if option in PM_aid.name["dephase"]:
            n_vz = 3 # <<< x, y, z
        elif option in PM_aid.name['emission']:
            n_vz = 2
        else:
            raise NameError( PM_aid.mes['NameError'] )
        self.n_actions = n_vz
        self.n_states = 8  # State: rho[i,j]
        self.counter = 0  # <<< mark the *position in the playground
        self.dw = dw
        self.time = T
        self.option = option # choose which type of system we want to use ('dephase' or 'emission')
        #self.note = open('Log//theta%s log.dat'%(str(theta/np.pi).replace('.', '')),'w')
        # Try a control group without control
        self.horizon = self._init_datum()

    def _init_datum(self):
        baseline = np.arange(self.interval+1, dtype=float)
        self.Vk = np.zeros_like(self.Vk) # turn-off control
        self.simple_control(self.dw, self.time, option=self.option, _ideal=False) # prepare rho_0. rho_1 list
        for i in range(self.interval):
            t = (i+1)*self.tau
            rho0, rho1 = self.rho[:,i+1]
            baseline[i+1] = Eval.qfisher2(rho0,rho1,self.dw)/t
        return baseline

    def reset(self):
        self.counter = 0
        rho_fn0 = qt.ket2dm(self.st1)
        observation = self._translateRHO(rho_fn0)
        #self.note.write(str(self.counter) + ': ' + 'INI' + ' ----- '+'INI'+'\n')
        #self.Vk = np.zeros_like(self.Vk)  # Vk: [3, interval]; Vz(t): [2, :]
        # self.note = open('log.txt','w')
        return observation  # state in the first time interval

    def _translateN_V(self, _action):
        # we restrict _action field to [-maxvk, maxvk]
        _action = np.clip(_action, -self.maxvk, +self.maxvk)#.astype(np.float32)
        # maybe need to translate action to Vk
        #_action *= 0.5
        return _action

    def _translateRHO(self, _rho):
        ket = np.zeros((8))
        _rho = _rho.data.toarray()
        ket[0], ket[1] = _rho[0, 0].real, _rho[0, 0].imag #  - 0.5
        ket[2], ket[3] = _rho[0, 1].real, _rho[0, 1].imag
        ket[4], ket[5] = _rho[1, 0].real, _rho[1, 0].imag
        ket[6], ket[7] = _rho[1, 1].real, _rho[1, 1].imag #  - 0.5
        return ket #* 10

    def _translateQFI(self, _t, _qfi):
        # fine tuning the qfi to reward; in principle, 
        # later qfi have larger reward?
        no_control = self.horizon[self.counter+1] #!!!!!! # ideally no control
        
        r = ( _qfi - no_control * 1.001 ) / no_control * 10 #   # r<0 can has information too !
        
        if self.counter == self.interval-1:
            r = r * 10 
            # equation r = ( _qfi - no_control) / no_control is enough for obtaining the 
                            # result better than no control. Thus, this part is to magnify the effect of
                            # final QFI to eliminate the possibility that the path of the higher QFI
                            # might cross the QFI lower than without control
            #else:
            #    r = -10
        return r

    def step(self, action):
        done = False
        
        # self.counter < self.interval:  # <<< in the playground
        new_Vk = self._translateN_V(action)
        self.Vk[:self.n_actions, self.counter] = new_Vk

        # calculate evolve to current time
        pre_time = self.counter * self.tau
        cur_time = (self.counter+1) * self.tau

        rho0, rho1 = self.continue_control(self.dw, pre_time, cur_time, option=self.option, _ideal=False)

        observation = self._translateRHO(rho0)  # translate to next state(t'=t+1) i.e. self.rho[0,self.counter+1]
        #sld, rho_ave = Eval.tedious_sld(rho0, rho1, self.dw)
        #qfi_step = Eval.qfisher(rho_ave,sld)/cur_time    # don't waste, store one qfi
        qfi_step = Eval.qfisher2(rho0,rho1,self.dw)/cur_time

        # reward from quantum fisher information at
        reward = 0
        reward += self._translateQFI(cur_time, qfi_step)

        # reach the last time interval
        if self.counter == self.interval-1:
            done = True
        # tell agent to tune control field in the next time slice
        self.counter += 1

        return observation, reward, done, qfi_step  # <<< How to deal with the end node
