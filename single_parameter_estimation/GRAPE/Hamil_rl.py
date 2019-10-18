import qutip as qt
import numpy as np
import Eval
import PM_aid # For error message


class PhysModel:

    def __init__(self, theta, phi, gamma, w0, T, tau ):

        self.time = T
        self.interval = round( float(T/tau) )
        Vk1 = 0.5 - np.random.rand(self.interval)  # init GRAPE control
        Vk2 = 0.5 - np.random.rand(self.interval)
        Vk3 = 0.5 - np.random.rand(self.interval)
        self.Vk = np.vstack([Vk1, Vk2, Vk3])  # [xyz,time]
        #self.Vk = np.zeros_like(self.Vk)
        self.H = np.array([qt.qeye(2) for i in range(self.interval)])  # init H
        self.rho = np.array([[qt.qeye(2) for i in range(self.interval + 1)]
                                , [qt.qeye(2) for i in range(self.interval + 1)]])  # init density matrix
        self.Hph = np.array([qt.qeye(2) for i in range(3)])  # init dephase (left option: time dependent)
        self.init_para(theta, phi, gamma, tau, w0) # <<< feed parameter
        self.tlist = np.arange(0, T + tau, tau)

    def init_para(self, theta, phi, gamma, tau, w0):
        # >>> quantum operator
        self.tau=tau
        self.w0 = w0
        self.theta = theta
        self.phi = phi
        self.gamma = gamma
        smx = qt.tensor(qt.sigmax())
        smy = qt.tensor(qt.sigmay())
        smz = qt.tensor(qt.sigmaz())
        self.smv = np.array([smx, smy, smz])
        self.init_state()
        a = np.sin(self.theta) * np.cos(self.phi)  # vector n=(nx,ny,nz)
        b = np.sin(self.theta) * np.sin(self.phi)
        c = np.cos(self.theta)
        self.nv = np.array([a, b, c])
        self.smn = np.dot(self.smv, self.nv)  # sigma*n
        self.smplus = (smx + 1j*smy) / 2.0    # sigma^+
        self.sminus = (smx - 1j*smy) / 2.0    # sigma^-

        # >>> Hamiltonian
        # H0
        self.dxh0 = 0.5 * self.smv[-1]  # smz
        
        # >>> store master equation options
        # option in PM_aid.name["dephase"]:
        self.Hph[0] = np.sqrt(0.5 * gamma) * self.smn
        # option in PM_aid.name["emission"]:
        gamma_plus = 0; gamma_minus = gamma  #  PROBLEM SETTING; or changed
        self.Hph[1] = np.sqrt(gamma_plus) * self.smplus
        self.Hph[2] = np.sqrt(gamma_minus) * self.sminus

    def update_h(self, w0, evlove_interval):
        # update the Hamiltonian when apply new control field
        # on a new interval; >>> "continous_control"
        # H0 + Vk
        for j in evlove_interval:
            H = (
                    w0 * self.dxh0 +
                    np.dot(self.Vk[:, j], self.smv)
                )
            self.H[j] = H  # + self.Hph[j+1]
        
    def init_state(self):
        # >>> initial state
        self.st0 = (qt.basis(2, 0)).unit()

        self.st1 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()

        self.st2 = (qt.basis(2, 0) - qt.basis(2, 1)).unit()

        self.POVM = [qt.ket2dm(self.st1), qt.ket2dm(self.st2)]

        # >>> initial density matrix
        st = self.st1
        rho_fn0 = qt.ket2dm(st)
        self.store_rho(0, rho_fn0, 0)
        rho_fn1 = qt.ket2dm(st)
        self.store_rho(1, rho_fn1, 0)

    def store_rho(self, tag, a, j):

        self.rho[tag, j] = a

    def simple_control(self, _dw, tt, option="D", _ideal=True):

        # 'simple gra, calculate full evolution with two parameters '
        # 'with control \n'
        # 'theta=%.2f; dephasing(%s)gamma=%.1f'%(_theta,_ideal,_gamma)

        # >>> init state
        rho_fn0 = self.rho[0,0] + 0
        rho_fn1 = self.rho[1,0] + 0
        
        interval_ = round( float(tt / self.tau) )  # evolve to tt<=T
        full_evolve = np.arange(interval_)
        # >>> w0
        self.update_h(self.w0 - _dw/2, full_evolve) # >>> reference case

        # solve master equation
        if _ideal:
            c_ops = []  # no dephasing
        else:
            if option in PM_aid.name["dephase"]:
                c_ops = [self.Hph[0]]  # dephasing, time independent
            elif option in PM_aid.name["emission"]:
                c_ops = [self.Hph[1], self.Hph[2]]  # emission, time independent
            else:
                raise NameError( PM_aid.mes['NameError'] )
        for i in full_evolve:
            medata = qt.mesolve(self.H[i], rho_fn0, np.linspace(0, self.tau, 11), c_ops, [])
            rho_fn0 = medata.states[-1]
            self.store_rho(0, rho_fn0, i + 1)

        # >>> w0+dw
        self.update_h(self.w0 + _dw/2, full_evolve)  # >>> shift case; update Hamilaonian

        # solve master equation
        if _ideal:
            c_ops = []  # no dephasing
        else:
            if option in PM_aid.name["dephase"]:
                c_ops = [self.Hph[0]]  # dephasing, time independent
            elif option in PM_aid.name["emission"]:
                c_ops = [self.Hph[1], self.Hph[2]]  # emission, time independent
            else:
                raise NameError( PM_aid.mes['NameError'] )
        for i in full_evolve:
            medata = qt.mesolve(self.H[i], rho_fn1, np.linspace(0, self.tau, 11), c_ops, [])
            rho_fn1 = medata.states[-1]
            self.store_rho(1, rho_fn1, i + 1)

        return rho_fn0, rho_fn1
    
    def continue_control(self, _dw, t0, t1, option="D", _ideal=True):

        # 'calculate temperal evolution with two parameters '
        # 'with control \n'
        #  in a new time interval
        
        # >>> init state at t=t0
        init_idx = round( float(t0 / self.tau) ) # >>> restore state from time t0
        rho_fn0 = self.rho[ 0, init_idx ] + 0
        rho_fn1 = self.rho[ 1, init_idx ] + 0

        interval_ = round( float((t1-t0) / self.tau) )  # evolve to tt<=T
        evlove_interval = init_idx + np.arange(interval_) # >>> continue evolve to t1
        #print(evlove_interval)

        # >>> w0; prepare the new Hamiltonian in [t0, t1]
        # init_idx>=0:
        self.update_h(self.w0 - _dw/2, evlove_interval)  # >>> reference case
        
        # solve master equation
        if _ideal:
            c_ops = []  # no dephasing
        else:
            if option in PM_aid.name["dephase"]:
                c_ops = [self.Hph[0]]  # dephasing, time independent
            elif option in PM_aid.name["emission"]:
                c_ops = [self.Hph[1], self.Hph[2]]  # emission, time independent
            else:
                raise NameError( PM_aid.mes['NameError'] )
        for i in evlove_interval:
            medata = qt.mesolve(self.H[i], rho_fn0, np.linspace(0, self.tau, 11), c_ops, [])
            rho_fn0 = medata.states[-1]
            self.store_rho(0, rho_fn0, i + 1)

        # >>> w0+dw; initialize the Hamiltonian at T=0
        # init_idx>=0:
        self.update_h(self.w0 + _dw/2, evlove_interval)  # >>> shift case; update Hamilaonian
        
        # solve master equation
        if _ideal:
            c_ops = []  # no dephasing
        else:
            if option in PM_aid.name["dephase"]:
                c_ops = [self.Hph[0]]  # dephasing, time independent
            elif option in PM_aid.name["emission"]:
                c_ops = [self.Hph[1], self.Hph[2]]  # emission, time independent
            else:
                raise NameError( PM_aid.mes['NameError'] )
        for i in evlove_interval:
            medata = qt.mesolve(self.H[i], rho_fn1, np.linspace(0, self.tau, 11), c_ops, [])
            rho_fn1 = medata.states[-1]
            self.store_rho(1, rho_fn1, i + 1)

        return rho_fn0, rho_fn1

if __name__ == "__main__":
    TIME = 5
    TAU = 0.1
    THETA = np.pi/4 # <<<
    PHI = 0
    DGAMMA = 0.1
    W0 = 1
    DW = 0.0001
    model_para = {"theta": THETA,
                  "phi": PHI,
                  "gamma": DGAMMA, 
                  "w0": W0,
                  "T": TIME,
                  "tau": TAU
                  }
    
    model = PhysModel(**model_para)
    model.Vk = np.zeros_like(model.Vk)
    model.Vk[2,:] = np.ones_like(model.Vk[2,:]) * -0.345
    rho0, rho1 = model.simple_control(DW, TIME, option='d', _ideal=False)
    #print(rho0, rho1)
    print(Eval.qfisher2(rho0, rho1, DW)/TIME)
    a = model.rho + 0
    import time
    tti = time.time()
    for i in range(round(float(TIME/TAU))):
        t0 = i*TAU
        t1 = t0 + TAU
        rho0, rho1 = model.continue_control(DW, t0, t1, option='e', _ideal=False)
    print(time.time()-tti,'s')
    #print(rho0, rho1)
    print(Eval.qfisher2(rho0, rho1, DW)/TIME)
    b = model.rho + 0
    #print(a-b)
    qfi_a =list([0]); qfi_b = list([0])
    import matplotlib.pyplot as plt
    for i in range(model.interval):
        t = (i+1)*TAU
        rho0, rho1 = a[:,i+1]
        qfi_a.append(Eval.qfisher2(rho0,rho1,DW)/t)
        rho0, rho1 = b[:,i+1]
        qfi_b.append(Eval.qfisher2(rho0,rho1,DW)/t)
    
    plt.plot(model.tlist, qfi_a, linewidth=5)
    plt.plot(model.tlist, qfi_b,'--')