#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils import set_init

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hidn_dim = 200
        
        self.layer00 = nn.Linear(s_dim, self.hidn_dim)
        self.layer01 = nn.Linear(self.hidn_dim, self.hidn_dim)
        self.layer02 = nn.Linear(self.hidn_dim, self.hidn_dim)
        
        self.layer10 = nn.Linear(self.hidn_dim, self.hidn_dim)

        #self.lstmc = nn.LSTMCell(self.hidn_dim, self.lstm_dim)
        #self.lstmc.bias_ih.data.fill_(0)
        #self.lstmc.bias_hh.data.fill_(0)
        self.actor0 = nn.Linear(self.hidn_dim, self.hidn_dim)
        self.actor_mu = nn.Linear(self.hidn_dim, a_dim)
        self.actor_sigma = nn.Linear(self.hidn_dim, a_dim)

        self.critic0 = nn.Linear(self.hidn_dim, self.hidn_dim)
        self.critic_value = nn.Linear(self.hidn_dim, 1)

        set_init([self.layer00, self.layer01, self.layer02, self.layer10, self.critic0, self.actor0, 
                  self.actor_mu, self.actor_sigma, self.critic_value])

    def forward(self, x):
        layer00_x = F.relu(self.layer00(x))
        layer01_x = F.relu(self.layer01(layer00_x))
        layer02_x = F.relu(self.layer02(layer01_x))
        
        layer10_x = F.relu(self.layer10(layer02_x))

        #layer02_x = layer02_x.view(-1, self.hidn_dim)
        #self.lstm_hidden = self.lstmc(layer02_x, self.lstm_hidden)
        #lstm_out = layer02_x # self.lstm_hidden[0]
        #layer02_x

        actor_x = F.relu(self.actor0(layer10_x))
        mu = 0.5 * F.softshrink(self.actor_mu(actor_x), lambd=0.25) #0.25
            # soften the output of the action value
            # saturate the activation of non-zero action
        sigma = F.softplus(self.actor_sigma(actor_x)) + 0.00001     # avoid 0

        critic_x =  F.relu(self.critic0(layer10_x))
        values =  self.critic_value(critic_x)
        return mu, sigma, values
