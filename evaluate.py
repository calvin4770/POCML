import torch
import torch.nn.functional as F

from dataloader import GraphEnv, Env
from model import POCML
from tqdm import tqdm

def accuracy(model, dataloader):
    total, correct = 0, 0
    confidences = torch.tensor([])

    for trajectory in dataloader:
        model.init_time()

        oh_o_first = F.one_hot(trajectory[:,0,0], num_classes=model.n_obs).to(torch.float32)

        # model.init_state(state_idx=trajectory[0,0,4])       
        model.init_state(obs = oh_o_first)                  #  treat the first observation as the spacial case. 
        #model.update_memory(model.u, oh_o_first, lr=self.lr_M, reg_M=self.reg_M, max_iter=self.max_iter_M, eps=self.eps_M)        #  memorize the first observation
        for t in range(trajectory.shape[1]):
            o_pre = trajectory[:, t, 0]
            a = trajectory[:, t, 1]
            o_next = trajectory[:, t, 2]

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
            c = torch.einsum("bi,bi->b", oh_o_next_pred, oh_o_next)
            confidences = torch.cat([confidences, c], dim=0)
            correct += (torch.argmax(oh_o_next_pred, dim=1) == torch.argmax(oh_o_next, dim=1)).sum().item()
            total += oh_o_next.shape[0]
    return correct / total, confidences.tolist()

def benchmark_accuracy(model, dataset):
    total, correct = 0, 0
    confidences = []

    with torch.no_grad():
        for x, y in dataset:
            model.reset_state()
            y_pred = model(x)[1:, :]
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
                pred_state = pred_state.unsqueeze(0)
                ps = model.get_expected_state(pred_state, in_place=False).squeeze()
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
                       lr=1,
                       max_iter=1,
                       eps=1e-3,
                       beta=1000,
                       test_acc=True):
    total, correct = 0, 0
    for t1, t2 in tqdm(dataset):
        traj1 = t1.unsqueeze(0)
        if test_acc:
            traj2 = t2.unsqueeze(0)
        else:
            traj2 = t1.unsqueeze(0)
        
        model.init_time()
        model.init_memory()
        model.init_state(state_idx=traj1[:, 0, 3])
        model.traverse(
            traj1,
            update_state_given_obs=False,
            update_memory=True,
            softmax=softmax,
            lr=lr,
            max_iter=max_iter,
            eps=eps,
            beta=beta
        ) # populate memory

        model.init_time()
        model.init_state(state_idx=traj2[:, 0, 3])
        y_pred = model.traverse(
            traj2, 
            update_state_given_obs=update_state_given_obs, 
            update_memory=update_memory,
            softmax=softmax,
            lr=lr,
            max_iter=max_iter,
            eps=eps,
            beta=beta
        ) # one-hot

        y = traj2[0, :, 2] # not one-hot
        correct += (y == y_pred.argmax(dim=1)).sum().item()
        total += y.shape[0]
    return correct / total

def zero_shot_accuracy_benchmark(model, dataset, test_acc=True):
    total, correct = 0, 0
    for t1, t2, y in dataset:
        traj1 = t1
        if test_acc:
            traj2 = t2
        else:
            traj2 = t1
        
        model.reset_state()
        model(traj1) # populate memory
        y_pred = model(traj2)[1:, :]
        correct += (y == y_pred.argmax(dim=1)).sum().item()
        total += y.shape[0]
    return correct / total

def test_two_tunnel(model: POCML):
    trajectory_length, num_desired_trajectories = 15, 1
    env = GraphEnv(
        env='two tunnel', 
        trajectory_length=trajectory_length, 
        num_desired_trajectories=num_desired_trajectories
    )
    env.populate_graph_preset()
    traj = torch.tensor(
        [[6, 0, 3, 8, 5],
        [3, 0, 2, 5, 2],
        [2, 2, 1, 2, 1],
        [1, 2, 0, 1, 0],
        [0, 1, 3, 0, 3],
        [3, 1, 5, 3, 6]]
    ).unsqueeze(0)
    model.init_time()
    model.init_memory()
    model.init_state(state_idx=traj[:, 0, 3])
    model.traverse(
        traj,
        update_state_given_obs=False,
        update_memory=True,
        softmax=False,
        lr=5,
        max_iter=1,
        eps=1e-3,
        beta=100,
        debug=True)

    mask = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 0, 1]])

    ps = []
    e = Env(env)
    e.init_state(5)
    model.init_state(obs=e.get_obs().unsqueeze(0), mask=mask)
    ps.append(model.u)
    policy = [1, 3, 3, 1, -1, 0, -1, -1, 0]
    for _ in range(20):
        state_est = model.u.argmax().item()
        a = policy[state_est]
        print("action", a)
        e.step(a)

        oh_a = F.one_hot(torch.tensor(a), num_classes=4).to(torch.float32).unsqueeze(0)
        hd_s_pred_bind_precleanup = model.update_state(oh_a) # update state by binding action
        oh_u_next = model.get_expected_state(hd_s_pred_bind_precleanup) # get p(u | \phi(\hat{s}_{t+1}'))
        oh_o_next = e.get_obs().unsqueeze(0)
        print("obs", e.get_obs())
        model.update_state_given_obs(oh_o_next, mask=mask)
        model.clean_up()
        ps.append(model.u)

        if e.state == 6:
            print("goal state reached")
            break
    return torch.cat(ps, dim=0).T