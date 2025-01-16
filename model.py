import torch
import torch.nn.functional as F
import numpy as np

# Computes the similarity between random feature vectors.
# Input shape:
# x: [out_dim]; y: [out_dim] OR
# x: [out_dim, B]; y: [out_dim] OR
# x: [out_dim]; y: [out_dim, B] OR
# x: [out_dim, B1]; y: [out_dim, B2]
def sim(x, y, eps=1e-6):
    relu = lambda x: torch.maximum(x, torch.ones_like(x) * eps)
    D = x.shape[0]
    if len(x.shape) == 1 and len(y.shape) == 1:
        return relu((x @ y.conj()).real / D)
    elif len(x.shape) == 2 and len(y.shape) == 1:
        return relu(torch.einsum("ji,j->i", x, y.conj()).real / D)
    elif len(x.shape) == 1 and len(y.shape) == 2:
        return relu(torch.einsum("j,ji->i", x, y.conj()).real / D)
    else:
        return relu(torch.einsum("ir,ic->rc", x, y.conj()).real / D)

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
                 memory_bypass=False,
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
    def init_state(self, obs=None, state_idx=None):
        if state_idx is not None:
            self.u = torch.zeros(self.n_states)
            self.u[state_idx] = 1
        else:
            if obs is None:
                self.u = torch.ones(self.n_states) / self.n_states
            else:
                self.u = self.get_state_from_memory(obs)
        self.clean_up()

    # Initialize empty memory, with the option to pass in pre-existing memory.
    # memory = (M, state_counts)
    # eps to make sure state counts can be normalized
    def init_memory(self, memory=None, eps=1e-3):
        if self.memory_bypass:
            self.M = torch.nn.Parameter(10 * torch.eye(self.n_states), requires_grad=False)
        else:
            if memory is None:
                self.M = torch.nn.Parameter(torch.zeros(self.n_obs, self.n_states), requires_grad=False)
            else:
                self.M = torch.nn.Parameter(memory[0], requires_grad=False)

    def update_memory(self, u, x, lr=0.1):
        ps = self.__prob_obs_given_state()
        dM = torch.einsum("j,ij->ij", u, x.unsqueeze(1) - ps)
        self.M += lr * dM

    # Retrieves state from memory given obs (Eq. 22).
    def get_state_from_memory(self, x):
        p_x_given_u = self.__prob_obs_given_state().T @ x
        return p_x_given_u / p_x_given_u.sum()

    def __prob_obs_given_state(self):
        return F.softmax(self.M, dim=0)

    # Retrieves obs from memory given state (Eq. 21).
    # input shape: [n_s] OR [..., n_s]
    def get_obs_from_memory(self, u):
        return self.__prob_obs_given_state() @ u

    # returns \hat{u}_t given \phi(\hat{s}_t')
    def get_expected_state(self, state, in_place=True):
        phi_Q = self.get_state_kernel()
        sims = sim(phi_Q, state)
        u = sims / sims.sum()
        if in_place:
            self.u = u
        return u
    
    # Update state given action (pre clean-up) (Eq. 18).
    def update_state(self, action):
        v = self.V @ action
        phi_v = self.random_feature_map(v)
        self.state = self.state * phi_v
        return self.state
    
    def update_state_given_obs(self, oh_o_next, eps=1e-4):
        x_given_u = self.get_obs_from_memory(torch.eye(self.n_states)) @ oh_o_next
        u = (self.u + eps) * (x_given_u + eps)
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

    def traverse(self, traj, update_state_given_obs=False, update_memory=True, softmax=False, beta=1000, debug=False):
        oh_o_first = F.one_hot(traj[0,0], num_classes=self.n_obs).to(torch.float32)
        if update_memory:
            if debug:
                print(self.u, oh_o_first)
            self.update_memory(self.u, oh_o_first)

        predictions = []
        for _, a, o_next, _, _ in traj:
            oh_o_next = F.one_hot(o_next, num_classes=self.n_obs).to(torch.float32)  # one-hot encoding of the first observation
            oh_a = F.one_hot(a, num_classes=self.n_actions).to(torch.float32)     # one-hot encoding of the first observation
            
            hd_s_pred_bind_precleanup = self.update_state(oh_a) # update state by binding action
            oh_u_next = self.get_expected_state(hd_s_pred_bind_precleanup) # get p(u | \phi(\hat{s}_{t+1}'))
            if softmax:
                oh_u_next = F.softmax(beta * oh_u_next, dim=0)
            oh_o_next_pred = self.get_obs_from_memory(oh_u_next) # predict observation with updated state
            predictions.append(oh_o_next_pred)

            # Clean up state \phi(\hat{s}_{t+1})
            if update_state_given_obs:
                self.update_state_given_obs(oh_o_next) # set u_{t+1} to p(u_{t+1} | s_{t+1}, x_{t+1} )
            self.clean_up()
            if update_memory:
                if debug:
                    print(oh_u_next, oh_o_next)
                self.update_memory(oh_u_next, oh_o_next)
        return torch.stack(predictions, dim=0)

class LSTM(torch.nn.Module):
    def __init__(self, n_obs, n_actions, n_states, hidden_size, include_init_state_info=True):
        super().__init__()
        if include_init_state_info:
            in_dim = n_obs + n_actions + n_states
        else:
            in_dim = n_obs + n_actions
        out_dim = n_obs
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(in_dim, hidden_size, 1, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, out_dim)

        self.reset_state()

    def reset_state(self):
        self.h, self.c = torch.zeros(1, self.hidden_size).to(torch.float32), torch.zeros(1, self.hidden_size).to(torch.float32)

    # state = initial state in traj
    def forward(self, x):
        # x has shape [L, in_dim]
        out, (self.h, self.c) = self.lstm(x, (self.h, self.c)) # [L, hidden_size]
        y = self.fc(out)  # [L, out_dim]
        return y
    
class Transformer(torch.nn.Module):
    def __init__(self, n_obs, n_actions, n_states, d_model, n_heads, hidden_size, include_init_state_info=True):
        super().__init__()
        if include_init_state_info:
            in_dim = n_obs + n_actions + n_states
        else:
            in_dim = n_obs + n_actions
        out_dim = n_obs
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.model = torch.nn.TransformerEncoderLayer(d_model, n_heads, hidden_size, batch_first=True)
        self.fc1 = torch.nn.Linear(in_dim, d_model)
        self.fc2 = torch.nn.Linear(d_model, out_dim)

    def reset_state(self):
        pass

    # state = initial state in traj
    def forward(self, x):
        # x has shape [L, in_dim]
        out = self.model(self.fc1(x), 
                         src_mask=torch.nn.Transformer.generate_square_subsequent_mask(x.shape[0]), 
                         is_causal=True) # [L, hidden_size]
        y = self.fc2(out)  # [L, out_dim]
        return y