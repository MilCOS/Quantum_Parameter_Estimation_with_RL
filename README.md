## PyTorch implementation of "Generalizable control for quantum parameter estimation through reinforcement learning" 

### [arXiv:1904.11298](https://arxiv.org/abs/1904.11298) or [Full-text](https://www.nature.com/articles/s41534-019-0198-z)

**Single parameter estimation:**

1. source code that generate the "A3C+PPO" data in this paper.
2. some trained neural networks that have generalizability.

For example, the Hamiltonian with the dephasing dynamics,

<img src="http://latex.codecogs.com/gif.latex?\partial_t\hat{\rho}(t)=-i\left[\hat{H}(t),\hat{\rho}(t)\right]+\frac{\gamma}{2}\left[\hat{\sigma}_{\vec n}\hat{\rho}(t)\hat{\sigma}_{\vec n}-\hat{\rho}(t)\right]."/>

The control field to be optimized is,

<img src="http://latex.codecogs.com/gif.latex?\hat{H}(t)=\frac{1}{2}\omega_0\hat{\sigma}_3+{\vec u}(t)\cdot{\vec \sigma}."/>

The figure below shows the training procedure when the dephasing along the direction <img src="http://latex.codecogs.com/gif.latex?\vartheta=0.25\pi,~\phi=0"/>, the dephasing rate 0.1, and evolution time T=5.

![fig1](https://github.com/MilCOS/Quantum_Parameter_Estimation_with_RL/blob/master/fig/theta0.25dgamma0.10T5dt0.1.png)

3. implementation of GRAPE for quantum parameter estimation from this paper [PhysRevA.96.012117](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.012117).

