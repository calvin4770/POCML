import torch
import torch.nn.functional as F

from dataloader import GraphEnv, DataLoader
from model import POCML

def accuracy(model, dataloader):
    total, correct = 0, 0
    confidences = []

    for trajectory in dataloader:
        model.init_time()

        oh_o_first = F.one_hot(trajectory[0,0,0], num_classes=model.n_obs).to(torch.float32)

        # model.init_state(state_idx=trajectory[0,0,4])       
        model.init_state(obs = oh_o_first)                  #  treat the first observation as the spacial case. 
        #model.update_memory(model.state, oh_o_first)        #  memorize the first observation
        
        for o_pre, a, o_next, _, _ in trajectory[0]:

            oh_o_next = F.one_hot(o_next, num_classes=model.n_obs).to(torch.float32)  # one-hot encoding of the first observation
            oh_a = F.one_hot(a, num_classes=model.n_actions).to(torch.float32)     # one-hot encoding of the first observation
            
            hd_s_pred_bind_precleanup = model.update_state(oh_a) # update state by binding action
            oh_u_next = model.get_expected_state(hd_s_pred_bind_precleanup) # get p(u | \phi(\hat{s}_{t+1}'))
            oh_o_next_pred = model.get_obs_from_memory(oh_u_next) # predict observation with updated state

            # Clean up state \phi(\hat{s}_{t+1})
            model.update_state_given_obs(oh_o_next) # set u_{t+1} to p(u_{t+1} | s_{t+1}, x_{t+1} )
            model.clean_up()

            # Update memory
            #model.update_memory(oh_u_next, oh_o_next)

            model.inc_time()

            confidences.append(oh_o_next_pred[o_next].item())
            if oh_o_next[torch.argmax(oh_o_next_pred, dim=0)] == 1:
                correct += 1
            total += 1
    return correct / total, confidences

def benchmark_accuracy(model, dataset):
    total, correct = 0, 0
    confidences = []

    with torch.no_grad():
        for x, y in dataset:
            model.reset_state()
            y_pred = model(x)
            correct += (y.argmax(dim=1) == y_pred.argmax(dim=1)).sum()
            total += y.shape[0]
            confidences.append(torch.einsum("ij,ij->i", y, F.sigmoid(y_pred)))
        
    return (correct / total).item(), torch.cat(confidences).tolist()

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
                ps = model.get_expected_state(pred_state, in_place=False)
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

# dataset generated from gen_zero_shot_dataset in GraphEnv
# of the form [(exploration trajectory, testing trajectory)]
def zero_shot_accuracy(model, 
                       dataset, 
                       update_state_given_obs=True, 
                       update_memory=False,
                       softmax=False, 
                       beta=1000):
    total, correct = 0, 0
    for traj1, traj2 in dataset:
        #traj2 = traj1 # TODO
        model.init_time()
        model.init_memory(bias=False)
        model.init_state(state_idx=traj1[0, 3])
        model.traverse(traj1,
                       update_state_given_obs=False,
                       update_memory=True,
                       softmax=softmax,
                       beta=beta) # populate memory

        model.init_time()
        model.init_state(state_idx=traj2[0, 3])
        y_pred = model.traverse(
            traj2, 
            update_state_given_obs=update_state_given_obs, 
            update_memory=update_memory,
            softmax=softmax,
            beta=beta) # one-hot

        y = traj2[:, 2] # not one-hot
        correct += (y == y_pred.argmax(dim=1)).sum()
        total += y.shape[0]
    return correct / total

def zero_shot_accuracy_benchmark(model, dataset):
    total, correct = 0, 0
    for traj1, traj2, y in dataset:
        #traj2 = traj1 # TODO
        model.reset_state()
        model(traj1) # populate memory
        y_pred = model(traj2)
        correct += (y == y_pred.argmax(dim=1)).sum()
        total += y.shape[0]
    return correct / total

def test_two_tunnel(model: POCML):
    trajectory_length, num_desired_trajectories = 10, 1
    env = GraphEnv(
        env='two tunnel', 
        batch_size=trajectory_length, 
        num_desired_trajectories=num_desired_trajectories
    )

    train_dataset = env.gen_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    state_idx = env.dataset.start_nodes[0]
    traj = train_dataloader[0]

    model.init_time()
    model.init_memory(bias=False)
    model.init_state(state_idx=state_idx)
    model.traverse(traj)

    model.init_state