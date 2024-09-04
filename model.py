import torch
import torch.nn.functional as F
import numpy as np

# Computes the similarity between random feature vectors.
# Input shape:
# x: [out_dim]; y: [out_dim] OR
# x: [out_dim, B]; y: [out_dim] OR
# x: [out_dim]; y: [out_dim, B] OR
# x: [out_dim, B1]; y: [out_dim, B2]
def sim(x, y):
    D = x.shape[0]
    if len(x.shape) == 1 and len(y.shape) == 1:
        return F.relu((x @ y.conj()).real / D)
    elif len(x.shape) == 2 and len(y.shape) == 1:
        return F.relu(torch.einsum("ji,j->i", x, y.conj()).real / D)
    elif len(x.shape) == 1 and len(y.shape) == 2:
        return F.relu(torch.einsum("j,ji->i", x, y.conj()).real / D)
    else:
        return F.relu(torch.einsum("ir,ic->rc", x, y.conj()).real / D)

class RandomFeatureMap(torch.nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha                  # inverse length scale
        self.W = torch.nn.Parameter(torch.randn(out_dim, in_dim).to(torch.complex64) / alpha, requires_grad=False)
        #self.W = torch.nn.Parameter((torch.rand(out_dim, in_dim).to(torch.complex64) * 2 * np.pi / alpha))    # sinc kernel

    # Applies the random feature map.
    # Input shape: [in_dim]
    # Output shape: [out_dim], OR
    # Input shape: [in_dim, ...]
    # Output shape: [out_dim, ...]
    def forward(self, x):
        if len(x.shape) == 1:
            out = torch.exp(1j * self.W @ x.to(torch.complex64) / self.alpha)
            return out
        else:
            # n = torch.norm(x, p=2, dim=-1, keepdim=True)
            n = 1
            out = torch.exp(1j * torch.einsum("ij,j...->i...", self.W, (x/n).to(torch.complex64)) / self.alpha )
            # out = torch.exp(1j * ((x/n).to(torch.complex64) @ self.W.T) / self.alpha ) / self.sqrt_out_dim
            #return out/torch.norm(out, p=2, dim=-1, keepdim=True)
            return out

class POCML(torch.nn.Module):
    def __init__(self,
                 n_obs,
                 n_states,
                 n_actions,
                 state_dim,
                 random_feature_dim,
                 alpha=1,
                 beta_obs=1,
                 beta_state=1,
                 memory_bypass=False,
                 decay=0.9,
                 mem_reweight_rate=1.0,
                 memory=None,
                 obs=None
    ):
        super().__init__()
        self.n_obs = n_obs # number of obss
        self.n_states = n_states # number of states
        self.n_actions = n_actions # number of actions
        self.state_dim = state_dim # dimension of state space
        self.random_feature_dim = random_feature_dim # dimension of random feature map output
        self.memory_bypass = memory_bypass # whether to bypass memory; bypassed memory will directly use \phi(Q) as memory

        self.Q = torch.nn.Parameter(torch.randn(state_dim, n_states, dtype = torch.float32) / np.sqrt(state_dim), requires_grad=False)
        self.V = torch.nn.Parameter(torch.randn(state_dim, n_actions, dtype = torch.float32) / np.sqrt(state_dim), requires_grad=False)
        self.random_feature_map = RandomFeatureMap(state_dim, random_feature_dim, alpha=alpha)

        self.init_memory(memory=memory)
        self.init_state(obs=obs)

        self.t = 0

    def init_time(self):
        self.t = 0

    def inc_time(self):
        self.t += 1

    # Initialize state, with the option to pass in the first observation.
    def init_state(self, obs=None, fixed_start=False):
        if fixed_start:
            self.u = torch.zeros(self.n_states)
            self.u[0] = 1
        else:
            if obs is None:
                self.u = torch.ones(self.n_states) / self.n_states
            else:
                self.u = self.get_state_from_memory(obs)
        self.clean_up()

    # Initialize empty memory, with the option to pass in pre-existing memory.
    # eps > 0 to ensure we don't divide by zero when retrieving state/obs from memory
    def init_memory(self, memory=None, eps=1e-6):
        if not self.memory_bypass:
            if memory is None:
                self.M = torch.nn.Parameter(torch.randn(self.n_states, self.n_obs).abs() * eps + torch.eye(self.n_states), requires_grad=False)
            else:
                self.M = torch.nn.Parameter(memory, requires_grad=False)
        else:
            self.M = torch.nn.Parameter(torch.eye(self.n_states), requires_grad=False)
        self.state_counts = self.M.sum(dim=1)
        self.obs_counts = self.M.sum(dim=0)

    def update_memory(self, u, x):
        if not self.memory_bypass:
            self.state_counts += u
            self.obs_counts += x
            self.M += torch.outer(u, x)

    # Retrieves state from memory given obs (Eq. 22).
    def get_state_from_memory(self, x):
        return self.M @ (x / self.obs_counts)

    # Retrieves obs from memory given state (Eq. 21).
    # input shape: [n_s] OR [..., n_s]
    def get_obs_from_memory(self, u):
        if len(u.shape) == 1:
            return self.M.T @ (u / self.state_counts)
        else:
            return torch.einsum("ij,...j,j->...i", self.M.T, u, 1 / self.state_counts)

    # returns \hat{u}_t given \phi(\hat{s}_t')
    def get_expected_state(self, state, eps=1e-6):
        phi_Q = self.get_state_kernel()
        sims = sim(phi_Q, state) + eps # prevent div by 0
        self.u = sims / sims.sum()
        return self.u
    
    # Update state given action (pre clean-up) (Eq. 18).
    def update_state(self, action):
        v = self.V @ action
        phi_v = self.random_feature_map(v)
        self.state = self.state * phi_v
        return self.state
    
    def update_state_given_obs(self, oh_o_next, eps=1e-6):
        x_given_u = self.get_obs_from_memory(torch.eye(self.n_states)) @ oh_o_next
        u = (self.u + eps) * (x_given_u + eps)
        print("!!!", self.u, x_given_u, u)
        self.u = u / u.sum()
        return self.u

    def clean_up(self):
        phi_Q = self.get_state_kernel()
        self.state = phi_Q @ self.u.to(torch.complex64)
        return self.state

    # Given the goal state, return the utilities for all actions (Eq. 35).
    def get_utilities(self, state):
        p = self.compute_weights(self.state)
        return self.V.T @ (state - self.Q @ p)

    # Given the goal state, return the action index with highest utility.
    def get_action(self, state, affordance=None):
        u = self.get_utilities(state)
        if affordance is not None:
            u = u * affordance
        return torch.argmax(u), u
    
    # Plan adapted from CML
    # TODO use model memory instead of affordances
    # TODO: add weight connection to the environment
    def plan(self, start, goal, env, weight=False):
        a_record = []
        o_record = []
        loc = int(start)
        length = 0

        self.init_time()
        oh_start = F.one_hot(start, num_classes=self.n_obs).to(torch.float32)
        self.init_state(obs = oh_start)                  #  treat the first observation as the spacial case. 
        self.update_memory(self.state, oh_start)        #  memorize the first observation    

        for i in range(self.o_size):
            o_record.append(loc)
            if torch.argmax(self.get_obs_from_memory)==goal:
                if weight:
                    return length, o_record
                else:
                    return i, o_record
            action, utility = self.move_one_step(loc, goal, a_record, env.affordance[loc], 
                    env.action_to_node, env.node_to_action_matrix[loc], env, weight)
            a_record.append(action)

            ## TODO TODO Environment needs to be interactive to receive corect observation. 

            if weight:
                length += env.w_connection[o_record[-1],loc]
        if weight:
            return length, o_record
        else:
            return i, o_record
        
    # move one step adapted from CML
    # TODO use model memory instead of affordances
    def move_one_step(self, loc, goal, a_record, affordance, action_to_node,
                      next_node_to_action, env, weight=False):  
        affordance_vector = torch.zeros(self.a_size, device=self.device)
        affordance_vector[affordance] = 1
        if weight:    
            for a in affordance:
                a = a.item()
                affordance_vector[a]/=(env.w_connection[action_to_node[a][0],
                                                    action_to_node[a][1]])
        affordance_vector_fix = affordance_vector.clone()
        not_recommended_actions = a_record
        affordance_vector_fix[not_recommended_actions] *= 0.

        # delta = self.Q[:,goal]-self.Q[:,loc]
        # utility = (self.W@delta) * affordance_vector_fix
        # utility = self.get_utilities(self, goal)

        # if torch.max(utility)!=0:
        #     action_idx = torch.argmax(utility).item()
        # else:
        #     utility = (self.V.T@delta) * affordance_vector
        #     action_idx = torch.argmax(utility).item()

        oh_loc = F.one_hot(loc, num_classes=self.n_obs).to(torch.float32)
        action_idx, utility =  self.get_action(oh_loc)
        oh_a = F.one_hot(action_idx, num_classes=self.n_actions).to(torch.float32)

        self.update_state(oh_a)
        self.clean_up(self.state)   # as an option

        return action_to_node[action_idx][1].item(), action_idx
    
    def normalize_action(self):
        self.V /= torch.norm(self.V, p=2, dim=0, keepdim=True)
        return self.V
    
    # Stat. Analysis


    def get_action_differences(self):
        return ((self.V[:, :, None] - self.V[:, None, :]).norm(p=2, dim=0)).detach()
    
    def get_action_similarities(self):
        return (self.V.T @ self.V).detach()
    
    def get_action_kernel(self):
        phi_V = self.random_feature_map(self.V)
        return phi_V.detach()
    
    def get_state_differences(self):
        return ((self.Q[:, :, None] - self.Q[:, None, :]).norm(p=2, dim=0)).detach()

    def get_state_similarities(self):
        return (self.Q.T @ self.Q).detach()

    def get_state_kernel(self):
        phi_Q = self.random_feature_map(self.Q)
        return phi_Q.detach()

