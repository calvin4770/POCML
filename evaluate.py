import torch
import torch.nn.functional as F

def accuracy(model, dataloader, mem_cleanup_rate=1):
    total = 0
    correct = 0

    for trajectory in dataloader:
        model.init_time()

        oh_o_first = F.one_hot(trajectory[0,0,0], num_classes=model.n_obs).to(torch.float32)
                
        model.init_state(obs = oh_o_first)                  #  treat the first observation as the spacial case. 
        model.update_memory(model.state, oh_o_first)        #  memorize the first observation
        
        for o_pre, a, o_next in trajectory[0]:

            oh_o_pre = F.one_hot(o_pre, num_classes=model.n_obs).to(torch.float32)  # one-hot encoding of the first observation
            oh_o_next = F.one_hot(o_next, num_classes=model.n_obs).to(torch.float32)  # one-hot encoding of the first observation
            oh_a = F.one_hot(a, num_classes=model.n_actions).to(torch.float32)     # one-hot encoding of the first observation
            
            # weight computation before action; at time t
            w_hat = model.compute_weights(model.state)

            # update state by binding action at time t+1, s^{hat}^{prime}_{t+1}, eq (18)
            hd_s_pred_bind_precleanup = model.update_state(oh_a)

            # clean up updated state
            model.clean_up(hd_s_pred_bind_precleanup)
            hd_s_pred_bind = model.state            

            # predict observation using updated state
            oh_o_next_pred = model.get_obs_from_memory(hd_s_pred_bind)

            # infer state at time t+1 via memory, s^{tilde}_{t+1} eq (22)
            hd_state_pred_mem = model.get_state_from_memory(oh_o_next)

            # reweight states
            model.reweight_state(hd_state_pred_mem, c=mem_cleanup_rate)

            if oh_o_next[torch.argmax(oh_o_next_pred, dim=0)] == 1:
                correct += 1
            total += 1
    return correct / total

def state_transition_accuracy(model, env):
    pass