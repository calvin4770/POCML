import torch
import torch.nn.functional as F
import numpy as np

# Computes the similarity between random feature vectors.
# Input shape:
# x: [out_dim]; y: [out_dim] OR
# x: [B, out_dim]; y: [out_dim] OR
# x: [out_dim]; y: [B, out_dim] OR
# x: [B1, out_dim]; y: [B2, out_dim]
def sim(x, y):
    D = x.shape[-1]
    if len(x.shape) == 1 and len(y.shape) == 1:
        return (x @ y.conj()).real / D
    elif len(x.shape) == 2 and len(y.shape) == 1:
        return torch.einsum("ij,j->i", x, y.conj()).real / D
    elif len(x.shape) == 1 and len(y.shape) == 2:
        return torch.einsum("j,ij->i", x, y.conj()).real / D
    else:
        return torch.einsum("ri,ci->rc", x, y.conj()).real / D

class RandomFeatureMap(torch.nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha # inverse length scale
        self.W = torch.nn.Parameter(torch.randn(out_dim, in_dim) / alpha)

    # Applies the random feature map.
    # Input shape: [in_dim]
    # Output shape: [out_dim], OR
    # Input shape: [..., in_dim]
    # Output shape: [..., out_dim]
    def forward(self, x):
        if len(x.shape) == 1:
            return torch.exp(1j * self.W @ x / self.alpha)
        else:
            return torch.exp(1j * torch.einsum("ij,...j->...i", self.W, x) / self.alpha)

class POCML(torch.nn.Module):
    def __init__(self,
                 n_obs,
                 n_states,
                 n_actions,
                 state_dim,
                 random_feature_dim,
                 alpha=1,
                 beta=1,
                 memory=None,
                 obs=None
    ):
        super().__init__()
        self.n_obs = n_obs # number of obss
        self.n_states = n_states # number of states
        self.n_actions = n_actions # number of actions
        self.state_dim = state_dim # dimension of state space
        self.random_feature_dim = random_feature_dim # dimension of random feature map output
        self.beta = beta # temperature parameter for softmax

        self.Q = torch.nn.Parameter(torch.randn(state_dim, n_states) / np.sqrt(state_dim))
        self.V = torch.nn.Parameter(torch.randn(state_dim, n_actions) / np.sqrt(state_dim))
        self.random_feature_map = RandomFeatureMap(state_dim, random_feature_dim, alpha=alpha)

        self.init_memory(memory=memory)
        self.init_state(obs=obs)

    # Initialize state, with the option to pass in the first observation.
    def init_state(self, obs=None):
        phi_Q = self.random_feature_map(self.Q)
        if obs is None:
            self.state = phi_Q.mean(dim=0)
        else:
            self.state = self.get_state_from_memory(obs)

    # Initialize empty memory, with the option to pass in pre-existing memory.
    def init_memory(self, memory=None):
        if memory is None:
            self.M = torch.nn.Parameter(torch.zeros(self.state_dim, self.n_obs).to(torch.complex64))
        else:
            self.M = memory

    def update_memory(self, state, obs):
        self.M = torch.outer(state, obs) + self.M

    # Retrieves state from memory given obs (Eq. 22).
    def get_state_from_memory(self, obs):
        return self.M @ obs

    # Retrieves obs from memory given state (Eq. 21).
    def get_obs_from_memory(self, state):
        return F.softmax(self.beta * (self.M.conj().T @ state).real, dim=0)
    
    # Cleans up state to be a linear combination of the columns of \phi(Q) (Eq. 19).
    # Returns weights in linear combination.
    def clean_up(self, state):
        phi_Q = self.random_feature_map(self.Q)
        weights = F.softmax(self.beta * (phi_Q.conj().T @ state).real)
        new_state = phi_Q @ weights
        self.state = new_state
        return weights
    
    # Compute weight given a state
    def compute_weights(self, state):
        phi_Q = self.random_feature_map(self.Q)
        weights = F.softmax(self.beta * (phi_Q.conj().T @ state).real)
        return weights
    
    # Update state given action (pre clean-up) (Eq. 18).
    def update_state(self, action):
        v = self.V @ action
        phi_v = self.random_feature_map(v)
        self.state = self.state * phi_v
        return self.state
    
    # Given the goal state, return the utilities for all actions (Eq. 35).
    def get_utilities(self, state):
        phi_Q = self.random_feature_map(self.Q)
        p = F.softmax(self.beta * (phi_Q.conj().T @ state).real)
        return self.V.T @ (state - self.Q @ p)

    # Given the goal state, return the action index with highest utility.
    def get_action(self, state, affordance=None):
        u = self.get_utilities(state)
        if affordance is not None:
            u = u * affordance
        return torch.argmax(u)