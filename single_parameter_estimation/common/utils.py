#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn
import torch
import numpy as np
import torch.multiprocessing as mp
import os


# Global counter from Morvan
class Counter():
  def __init__(self):
    self.val = mp.Value('i', 0)
    self.lock = mp.Lock()

  def increment(self):
    with self.lock:
      self.val.value += 1

  def value(self):
    with self.lock:
      return self.val.value

def v_wrap(np_array, dtype=np.float32):
    """change data type"""
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def set_init(layers):
    """initialize layers"""
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.01)
        nn.init.constant_(layer.bias, 1)

def discount_with_dones(rewards, dones, gamma):
    """discounted reward. Able to deal with multiple episodes at once"""
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def file_path(fpath):
    return os.path.join(os.getcwd(), fpath)

def save_net(train_model, theta, tau, NO=0):
    tmp_path = os.path.join(os.getcwd(), 'Net/ppo theta %i tau %i NO %i'%(theta/np.pi*100, tau*10, NO))
    torch.save(train_model.state_dict(), tmp_path)

def load_net(train_model, theta, tau, NO=0, ppo=True):
    if ppo:
        tmp_path = os.path.join(os.getcwd(), 'Net/ppo theta %i tau %i NO %i'%(theta/np.pi*100, tau*10, NO))
    else:
        tmp_path = os.path.join(os.getcwd(), 'Net/theta %i tau %i NO %i'%(theta/np.pi*100, tau*10, NO))
    train_model.load_state_dict( torch.load(tmp_path) )
    print('train from %s'%tmp_path)
    return train_model















