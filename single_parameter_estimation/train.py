#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import os, json, time
import torch.multiprocessing as mp

from common.utils import v_wrap, file_path, discount_with_dones
from model import Net


def _gaussian():
    """gaussian distribution method"""
    return torch.distributions.Normal

def _ppo_clip_app(epsilon, r_t, a):
    """cliping the ppo ratio"""
    clip_r_t = torch.clamp(r_t, 1-epsilon, 1+epsilon)
    surr_t = torch.min(r_t*a, clip_r_t*a)
    #print(r_t*a,clip_r_t*a,'\n\n\n',surr_t)
    return surr_t

def _ppo_loss_func(ppo_step, args, model, old_m, s_batch, a_batch, disr_batch):
    """calculate the ppo loss for global model using local experiences in .batch
    old_m: old policy; 
    """
    total_loss = torch.zeros(1,1)
    s = s_batch
    a = a_batch
    V_t = disr_batch
    mu, sigma, values = model.forward(s)
    
    # critic loss
    advantage = V_t - values
    c_loss = advantage.pow(2) * args.td_weight

    # policy gain
    distribution = _gaussian()
    m = distribution(mu, sigma)

    log_prob = torch.sum( m.log_prob(a), dim=1)
    old_log_prob = torch.sum( old_m.log_prob(a), dim=1)
    
    r_t = torch.exp(log_prob - old_log_prob)
    surr = _ppo_clip_app(args.ppo_epsilon, r_t, advantage.detach() )
    # entropy exploration
    entropy = torch.sum( - 0.5 * ( torch.log(2*np.pi*sigma**2) + 1), dim=1)
    exp_v = surr + args.entropy_weight * entropy
    # maximize performance
    a_loss = - exp_v
    #if ppo_step >= 0:
    total_loss = (a_loss + c_loss).sum()
    #else:
    #    total_loss = (a_loss).sum()

    return total_loss

def _ppo_train(args, T, lopt, lnet, opt, gnet, mu_batch, sigma_batch, s_batch, a_batch, disr_batch):
    """main body of training
    T: current episode; lopt: local optimizer; lnet: local model; opt: global optimizer; gnet: global model
    mu_batch, sigma_batch, s_batch, a_batch, disr_batch: experience where s_batch used to recreat V(s).
    """
    distribution = _gaussian()
    old_m = distribution(v_wrap(np.array(mu_batch)),
                         v_wrap(np.array(sigma_batch))
                         )
    for ppo_step in range(args.ppo_N):
        lopt.zero_grad()
        loss = _ppo_loss_func(ppo_step,
                args, gnet, old_m,
                v_wrap(np.vstack(s_batch)),
                v_wrap(np.array(a_batch), dtype=np.float32)
                    if a_batch[0].dtype == np.float32 else v_wrap(np.vstack(a_batch)),
                v_wrap(np.array(disr_batch)[:, None])
                )
        # calculate local gradients with PPO loss and update local network
        loss.backward()
        if args.clip_gradient_norm:
            torch.nn.utils.clip_grad_norm_(lnet.parameters(), 
                                           max_norm = args.max_gradient_norm)
        lopt.step()


    if args.ppo_decay:
        # Linearly decay learning rate
        args.ppo_epsilon = max(args.ppo_epsilon * (args.max_ep - T.value()) / args.max_ep, 1e-32)

    # Transfers net parameters from shared model to thread-specific model
    lnet.load_state_dict(gnet.state_dict())
    return loss.data

def _choose_action(model, s, xyz):
    """Sampling in action space"""
    mu, sigma, _ = model.forward(s)
    move = torch.zeros(xyz)
    distribution = _gaussian()
    move[:] = distribution(mu, sigma).sample().view(-1)
    return move.numpy(), mu.detach().numpy(), sigma.detach().numpy()

# === Main training process: agents interacte with environments ===
def train(T, res_queue, args, opt, gnet, env, rep=0):
    mpid = mp.current_process().name
    # !!! === init random seed ===
    if not args.share_seed:
        np.random.seed(mpid+args.seed)
        torch.manual_seed(mpid+args.seed)
    else:
        None

    # === ===
    lname = '%s_NO%i_worker0%s' % (env.option, rep, mpid)
    lnet = Net(env.n_states, env.n_actions)
    lnet.load_state_dict(gnet.state_dict())
    lnet.train()
    note = open(file_path('Log/%s Vklog theta_pi%i.txt'%(lname, 100*env.theta/np.pi)), 'w')
    header = ['theta', env.theta, 'tau', env.tau, 'time', env.time, 'worker', mpid]
    note.write( " ".join(map(lambda x: str(x), header))+'\n') # add headline
    ## local optimizers
    RMSPparas = {"lr": args.lr,
                 "alpha": args.lr_decay,
                 "eps": 1e-8
                 }
    lopt = torch.optim.RMSprop(gnet.parameters(), **RMSPparas)
    Adamparas = {"lr": args.lr}
    lopt = torch.optim.Adam(gnet.parameters(), **Adamparas)
    ##
    total_step = 1
    _old_qfi = 0
    tsne_num = 0
    tstart = time.time()
    while T.value() < args.max_ep:
        state = env.reset()
        buffer_mu, buffer_sigma, buffer_action = [], [], []
        buffer_state, buffer_dones, buffer_reward = [], [], []
        ep_r = 0.
        vk_list = []
        for t in range(args.max_ep_step):
            skip_drama = args.update_iter
            action, mu, sigma = _choose_action(lnet, v_wrap(state), env.n_actions)
            state_, reward, done, ep_qfi = env.step(action)
            if t == args.max_ep_step - 1:
                done = True

            ep_r += reward
            buffer_mu.append(mu)
            buffer_sigma.append(sigma)
            buffer_action.append(action)
            buffer_state.append(state)
            buffer_dones.append(done)
            buffer_reward.append(reward)
                
            if total_step % skip_drama == 0 or done:  # update global and assign to local net
                # Discount/bootstrap off value fn
                _, _, last_value = lnet.forward(v_wrap(state_))
                if buffer_dones[-1] == 0:
                    discount_rewards = discount_with_dones(buffer_reward+[ last_value.detach().numpy()[0] ], buffer_dones+[0], args.gamma)[:-1]
                else:
                    discount_rewards = discount_with_dones(buffer_reward, buffer_dones, args.gamma)

                #  update local policy PPOly and assync
                total_loss = _ppo_train(args, T, lopt, lnet, opt, gnet, 
                                        buffer_mu, buffer_sigma, 
                                        buffer_state, buffer_action, 
                                        discount_rewards)
                
                # empty the buffer
                buffer_mu, buffer_sigma, buffer_action = [], [], []
                buffer_state, buffer_dones, buffer_reward = [], [], []
                # done and collect information
                T.increment()
                res_queue.put([ep_r, ep_qfi])

            state = state_
            total_step += 1
            
            # ==== Storing Data ====
            vk_list.append( (env._translateN_V(action)).tolist() )
            if done:
                # prepare data output
                # temp store net for every worker
                if ep_qfi > _old_qfi:
                    #a3cnet_name =  lname+'-mid-point-a3c_net_theta_pi%i.pkl'%(100*env.theta/np.pi)
                    #torch.save(lnet, file_path('Net/'+a3cnet_name) )  # save evaluation net        
                    vk_name =  lname+'-mid-point-vk_theta_pi%i.json'%(100*env.theta/np.pi)
                    vk_data = {'qfi':ep_qfi, 'theta':env.theta, 'tlist': env.tlist.tolist(), 'VK': vk_list}
                    with open( file_path('Vk/'+vk_name), 'w') as fl:  # save vk
                        fl.write( json.dumps(vk_data) )
                        fl.flush()
                        fl.close()

                    note.write( '\n# %f %i\n'%(ep_qfi, tsne_num) )
                    _ = [note.write("  ".join(map(lambda x: str(x), item))+'\n') for item in vk_list]
                    note.flush()
                    _old_qfi = ep_qfi # !!!!!!!!!!!

                tsne_num += 1  # record the route of the agent per 100 steps
                if tsne_num%200 == 0:
                    note.write( '\n# %f %i\n'%(ep_qfi, tsne_num) )
                    _ = [note.write( "  ".join(map(lambda x: str(x), item))+'\n') for item in vk_list]
                    note.flush()
                vk_list = []

                break

        nseconds = time.time()-tstart
        if T.value()%100==0: print(
                'episode:', T.value(), round(nseconds,2), 's', ' qfi:', np.mean(ep_qfi),
                'loss:',total_loss, 'worker id', mpid
                )
        # if ep_qfi >= 7.2: break # early cut-off?

    note.close()
    res_queue.put(None)
