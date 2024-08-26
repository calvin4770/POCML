import torch
import torch.nn.functional as F

def accuracy(model, dataloader):
    total, correct = 0, 0
    confidences = []

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
            model.reweight_state(hd_state_pred_mem)

            confidences.append(oh_o_next_pred[o_next].item())
            if oh_o_next[torch.argmax(oh_o_next_pred, dim=0)] == 1:
                correct += 1
            total += 1
    return correct / total, confidences

def state_transition_consistency(model, env):
    node_to_action_matrix = env.node_to_action_matrix
    n_nodes = node_to_action_matrix.shape[0]
    
    phi_Q = model.get_state_kernel()
    phi_V = model.get_action_kernel()
    Q = model.Q
    V = model.V

    total, correct = 0, 0
    confidences = []
    distance_ratios = []

    for src in range(n_nodes):
        for tgt in range(n_nodes):
            action_idx = node_to_action_matrix[src][tgt]
            if action_idx != -1:
                pred_state = phi_Q[:, src] * phi_V[:, action_idx]
                ps = model.compute_weights(pred_state)
                confidences.append(ps[tgt].item())
                if torch.argmax(ps) == tgt:
                    correct += 1
                total += 1

                state_src = Q[:, src]
                state_tgt = Q[:, tgt]
                state_pred = state_src + V[:, action_idx]
                state_pred_dist = torch.linalg.norm(state_tgt - state_pred).item()
                state_dist_total = torch.linalg.norm(state_tgt - state_src).item()
                distance_ratios.append(state_pred_dist / state_dist_total)
    return correct / total, confidences, distance_ratios