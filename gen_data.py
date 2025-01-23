import torch
import random
import numpy as np
import model
from dataloader import GraphEnv, DataLoader

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    n_nodes = 9
    batch_size = 64        # Note: in og CML trajectory length == batch_size; in POCML this should be decoupled
    n_obs = 9
    trajectory_length = 12  # numer of node visits in a trajectory
    num_desired_trajectories= 24 * batch_size

    env = GraphEnv( n_items=n_nodes,                     # number of possible observations
        env='grid', 
        trajectory_length=trajectory_length, 
        num_desired_trajectories=num_desired_trajectories, 
        device=None, 
        unique=True,                         # each state is assigned a unique observation if true
        args = {"rows": 3, "cols": 3}
    )

    train_dataset = env.gen_dataset()
    test_dataset = env.gen_dataset()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    set_random_seed(70)
    main()