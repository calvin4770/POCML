import torch
import random
import numpy as np
import model
from dataloader import GraphEnv

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    n_nodes = 9
    n_obs = 9
    trajectory_length = 16  # numer of node visits in a trajectory
    num_desired_trajectories= 30

    env = GraphEnv(
        n_items=n_nodes,                     # number of possible observations
        env='grid', 
        batch_size=trajectory_length, 
        num_desired_trajectories=num_desired_trajectories, 
        device=None, 
        unique=True,                         # each state is assigned a unique observation if true
        args = {"rows": 3, "cols": 3}
    )

    lstm = model.LSTM(env.n_items, env.n_actions, env.size, 10)

if __name__ == "__main__":
    set_random_seed(70)
    main()