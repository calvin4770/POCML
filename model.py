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
        self.alpha = alpha                  # inverse length scale
        self.W = torch.nn.Parameter(torch.randn(out_dim, in_dim).to(torch.complex64) / alpha, requires_grad=False)
        #self.W = torch.nn.Parameter((torch.rand(out_dim, in_dim).to(torch.complex64) * 2 * np.pi / alpha))    # sinc kernel

    # Applies the random feature map.
    # Input shape: [in_dim]
    # Output shape: [out_dim], OR
    # Input shape: [..., in_dim]
    # Output shape: [..., out_dim]
    def forward(self, x):
        if len(x.shape) == 1:
            out = torch.exp(1j * self.W @ x.to(torch.complex64) / self.alpha)
            return out
        else:
            # n = torch.norm(x, p=2, dim=-1, keepdim=True)
            n = 1
            out = torch.exp(1j * torch.einsum("ij,...j->...i", self.W, (x/n).to(torch.complex64)) / self.alpha )
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
        self.beta_obs = beta_obs # temperature parameter for predicting observation
        self.beta_state = beta_state # temperature parameter for predicting state (after clean up)
        self.memory_bypass = memory_bypass # whether to bypass memory; bypassed memory will directly use \phi(Q) as memory
        self.decay = decay # decay parameter for memory
        self.mem_reweight_rate = mem_reweight_rate

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
    def init_state(self, obs=None):
        phi_Q = self.get_state_kernel()
        if obs is None:
            self.state = phi_Q.mean(dim=0)
        else:
            self.state = self.get_state_from_memory(obs)

    # Initialize empty memory, with the option to pass in pre-existing memory.
    def init_memory(self, memory=None):
        self.item_count = 0
        if not self.memory_bypass:
            if memory is None:
                self.M = torch.nn.Parameter(torch.zeros(self.random_feature_dim, self.n_obs, dtype=torch.complex64), requires_grad=False)
            else:
                self.M = memory
        else:
            self.M = self.get_state_kernel()

    def update_memory(self, state, obs):
        self.item_count += 1
        if not self.memory_bypass:
            if self.decay == "adaptive":
                state_from_obs = self.get_state_from_memory(obs)
                s = sim(state_from_obs, state)
                print("s", s)
                self.M += (1 - s) * torch.outer(state, obs)
            else:
                self.M *= self.decay
                self.M += torch.outer(state, obs)
        else:
            self.M = self.get_state_kernel()

    # Retrieves state from memory given obs (Eq. 22).
    def get_state_from_memory(self, obs):
        return self.M @ obs.to(torch.complex64)

    # Retrieves obs from memory given state (Eq. 21).
    def get_obs_from_memory(self, state):
        score = self.get_obs_score_from_memory(state)
        return F.softmax(score, dim=0)
    
    def get_obs_score_from_memory(self, state):
        return self.beta_obs * sim(self.M.T, state)

    def get_state_score(self, state):
        phi_Q = self.get_state_kernel()
        return self.beta_state * sim(phi_Q.T, state)

    def clean_up(self, state):
        phi_Q = self.get_state_kernel()
        weights = self.compute_weights(state)
        new_state = phi_Q @ weights.to(torch.complex64)
        self.state = new_state
        return weights
    
    def reweight_state(self, state_from_mem):
        if self.mem_reweight_rate == "adaptive":
            state_score = self.get_state_score(self.state)
            state_mem_score = self.get_state_score(state_from_mem)
            state_entropy = self.__entropy_from_logits(state_score)
            state_mem_entropy = self.__entropy_from_logits(state_mem_score)
            c = state_entropy / (state_entropy + state_mem_entropy)
        else:
            c = self.mem_reweight_rate
        self.state = (1-c) * self.state + c * (state_from_mem)

    def __entropy_from_logits(self, logits):
        return -torch.sum(F.softmax(logits, dim=0) * F.log_softmax(logits, dim=0))
    
    # Compute weight given a state
    def compute_weights(self, state):
        score = self.get_state_score(state)
        weights = F.softmax(score, dim=0)
        return weights
    
    # Update state given action (pre clean-up) (Eq. 18).
    def update_state(self, action):
        v = self.V @ action
        phi_v = self.random_feature_map(v)
        self.state = self.state * phi_v
        return self.state
    
    def update_representations(self, dQ, dV, refactor_memory=False):
        if refactor_memory:
            weights = F.softmax(sim(self.get_state_kernel().T, self.M.T), dim=0)
        self.Q += dQ
        self.V += dV
        if refactor_memory:
            self.M = torch.nn.Parameter(
                self.get_state_kernel() @ weights.to(torch.complex64),
                requires_grad=False
            )


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
        phi_V = self.random_feature_map(self.V.T).T
        return phi_V.detach()
    
    def get_state_differences(self):
        return ((self.Q[:, :, None] - self.Q[:, None, :]).norm(p=2, dim=0)).detach()

    def get_state_similarities(self):
        return (self.Q.T @ self.Q).detach()

    def get_state_kernel(self):
        phi_Q = self.random_feature_map(self.Q.T).T
        return phi_Q.detach()
        
    def get_memory_kernel(self):
        return sim(self.M.T, self.M)

