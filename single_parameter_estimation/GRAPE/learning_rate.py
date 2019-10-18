"""
Learning rate strategy
"""
import numpy as np

def momentum_str(Dwold,grad,alpha,epsilon):
    """
    grad = dE/dw
    a = alpha; b = epsilon
    v(t) = a * v(t-1) - b * grad(t)
    Dw(t)=v(t) = a * Dw(t-1) - b * grad(t)
    """    
    
    Dwnew = alpha * Dwold - epsilon * grad
    
    return Dwnew

def momentum_Nesterov(Dwold,grad,alpha,epsilon):
    None
    
def init_weight(w):
    g = np.ones(w.shape)
    return g

def sep_ada(weight,grad_old,grad,delta=0.05):
    """
    g = weight
    Dw = epsilon * g * dE/dw
    
    if ( dE/dw(t) * dE/dw(t-1) )>0
    then g(t) = g(t-1) + delta
    else g(t) = g(t-1) * (1-delta)
    """
    x,y = weight.shape
    g = weight
    g_new = np.copy(g)
    for i in range(x):
        for j in range(y):
            condi = (grad_old[i,j]*grad[i,j]>0)
            condi_init = (grad_old[i,j]*grad[i,j]==0)
            if condi:
                g_new[i,j] = g[i,j] + delta
            elif condi_init:
                g_new[i,j] = g[i,j]
            else:
                g_new[i,j] = g[i,j] * (1-delta)

    return g_new