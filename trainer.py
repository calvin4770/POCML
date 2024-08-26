import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

from model import sim

class CMLTrainer:
    def __init__(self, model, train_loader, norm=False, optimizer=None, criterion=None, val_loader=None, device=None):
        
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_loader = val_loader
        self.device = device
        self.norm = norm

    def train(self, epochs = 10):

        loss_record = []
        for _ in tqdm(range(epochs), desc="Epochs"):
            loss_record += self.train_epoch() # Concatenate the list of losses
        return loss_record

    def train_epoch(self):

        model = self.model 
        device = self.device
        norm = self.norm

        with torch.no_grad(): # Turn off auto grad to train with custome update rule
            loss_record = []
            for trajectory in self.train_loader:
                o_pre, action, o_next = trajectory[0,:,0].to(device),\
                                        trajectory[0,:,1].to(device),\
                                        trajectory[0,:,2].to(device)
                identity = torch.eye(self.model.a_size).to(self.device)
                state_diff = model.Q[:,o_next]-model.Q[:,o_pre]

                prediction_error = state_diff - model.V[:,action]
                desired = identity[action].T
                
                # Core learning rules:
                model.Q[:,o_next] += -0.1 * prediction_error
                model.V[:,action] += 0.01 * prediction_error
                model.W += 0.01 * desired@state_diff.T
                if norm:
                    model.V.data = model.V / torch.norm(model.V, dim=0)

                loss = nn.MSELoss()(prediction_error, torch.zeros_like(prediction_error))
                loss_record.append(loss.cpu().item())

            return loss_record


class POCMLTrainer(CMLTrainer):
    def __init__(self,
                 model,
                 train_loader,
                 norm=False,
                 optimizer=None,
                 criterion=None,
                 val_loader=None,
                 device=None,
                 lr_Q_o = 1.,
                 lr_V_o = 0.1,
                 lr_Q_s = 0,
                 lr_V_s = 0,
                 lr_all = 0.32,
                 reset_every = 1, # reset every N trajectories
                 refactor_memory = False,
                 normalize = False):

        #super.__init__(model, train_loader, norm=False, optimizer=None, criterion=None, val_loader=None, device=None)
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_loader = val_loader
        self.device = device
        self.norm = norm

        # POCML model param for training
        self.beta_obs = model.beta_obs                          # observation prediction temperature
        self.beta_state = model.beta_state                      # state prediction temperature
        self.alpha = model.random_feature_map.alpha             # (inverse) lengscale of the RBF
        self.lr_Q_o = lr_Q_o
        self.lr_V_o = lr_V_o
        self.lr_Q_s = lr_Q_s
        self.lr_V_s = lr_V_s
        self.lr_all = lr_all
        self.normalize = normalize
        self.reset_every = reset_every
        self.refactor_memory = refactor_memory

    # # Create tensor reused in the update rule (30 - 33)
    # # the tensor U is of shape n_s * n_s * n where U[i,j,k] = \omega_{i,j,k} K_{i，j} (s^{hat}_j - s_i) 
    def __prep_update(self, w_tilde, w_hat, oh_a):
        
        model = self.model 
        Q = model.Q

        omega = torch.einsum('ki,j->ijk', w_tilde, w_hat)                 # omega_ijk = w_tilde_{ki} * w_hat_{j}

        v_t = model.V @ oh_a                                # v_t []
        s_hat = (Q.T) + v_t[None, :]                        # s_hat_j = (s_j + v_t)
        diff_s = (Q.T)[:, None, :] - s_hat[None, :, :]      # generate n_s * n_s * n tensor [s_i - s_hat_j]_i,j
        
        diff_s_squared_norm = torch.sum(diff_s ** 2, dim=-1)

        # Approximating K with phi(Q)
        # K = ((model.random_feature_map(model.Q.T)) @ (model.random_feature_map(s_hat).T)).real / D

        # Compute exact K 
        K = torch.exp(-self.alpha * diff_s_squared_norm)

        self.update_tensor = torch.einsum('j,ij,ijm->ijm', w_hat, K, diff_s)

        u = torch.eye(self.model.n_states).to(self.device)        # TODO double check if this can be optimized
        self.Z_q = -1 * torch.einsum('ik,ijm,in->kmn', w_tilde, self.update_tensor, u)
        self.Z_v = torch.einsum('ik,ijm,n->kmn', w_tilde, self.update_tensor, oh_a)

        self.W_q = -1 * torch.einsum('ijm,in->imn', self.update_tensor, u)
        self.W_v = torch.einsum('ijm,n->imn', self.update_tensor, oh_a)


    def train(self, epochs:int = 10) -> list:

        loss_record = []

        for _ in tqdm(range(epochs), desc="Epochs"):
            loss_record += self.train_epoch() # Concatenate the list of losses
        return loss_record
    
    ## Naming convention
    #    hd_, sa_, oh_/*: respective objects in HD space, state-action space, and ``one_hot" space
    #    a, o, s: action, observation, state (note that "u" in the paper is replaced with s; hd_s and as_s is used to disambiguate the domain)
    #    _bind, _mem: state inferences are made from binding (eq (18,19), denoted \hat) vs from memory (eq 22, \tilde)
    def train_epoch(self) -> list:

        with torch.no_grad():

            model = self.model
            device = self.device
            normalize = self.normalize
            criterion = nn.CrossEntropyLoss()
            loss_record = []

            # memory transfer option/reset rate + for decay design
            model.init_memory()                    # reset model's memory for new epochs        
            
            for tt, trajectory in enumerate(self.train_loader):

                model.init_time()
                
                if tt % self.reset_every == 0:             # memory have to option to reset per trajectory
                    model.init_memory()
                                                            # TODO memory should have the option to reset per graph instance

                oh_o_first = F.one_hot(trajectory[0,0,0], num_classes=model.n_obs).to(torch.float32)
                
                model.init_state(obs = oh_o_first)                  #  treat the first observation as the spacial case. 
                model.update_memory(model.state, oh_o_first)        #  memorize the first observation

                if tt == 0: 
                    print("Current Trajectory", trajectory[0])
                    print("initial state:", trajectory[0,0,0])
                    print("Print initial score", model.get_obs_score_from_memory(model.state))
                    print("Obs similarity\n", sim(model.M.T, model.M.T))
                    print("Action difference\n", model.get_action_differences())
                    print("State  difference\n", model.get_state_differences())

                # o_pre  is the observation at time t
                dQ_total = torch.zeros_like(model.Q)
                dV_total = torch.zeros_like(model.V)
                for ttt, (o_pre, a, o_next) in enumerate(trajectory[0].to(device)):
                    dQ, dV, loss = self.__one_time_step(model, o_pre, a, o_next, tt, ttt, criterion, normalize=normalize)
                    dQ_total += dQ
                    dV_total += dV
                    loss_record.append(loss.cpu().item())
                model.update_representations(dQ_total, dV_total, refactor_memory=self.refactor_memory)

        return loss_record
    
    def __one_time_step(self, model, o_pre, a, o_next, tt, ttt, criterion, normalize=False):
        if tt == 0 and ttt == 0: 
            print("Time", model.t)
        
        oh_o_pre = F.one_hot(o_pre, num_classes=model.n_obs).to(torch.float32)  # one-hot encoding of the first observation
        oh_o_next = F.one_hot(o_next, num_classes=model.n_obs).to(torch.float32)  # one-hot encoding of the first observation
        oh_a = F.one_hot(a, num_classes=model.n_actions).to(torch.float32)     # one-hot encoding of the first observation
        
        # weight computation of state at time t (before action); eq (31)
        w_hat = model.compute_weights(model.state)

        # update state by binding action at time t+1, s^{hat}^{prime}_{t+1}, eq (18)
        hd_s_pred_bind_precleanup = model.update_state(oh_a)

        # clean up updated state
        weights = model.clean_up(hd_s_pred_bind_precleanup)
        hd_s_pred_bind = model.state

        # predict observation with updated state
        oh_o_next_pred = model.get_obs_from_memory(hd_s_pred_bind)

        # infer state from observation at time t+1 via memory, s^{tilde}_{t+1} eq (22)
        hd_state_pred_mem = model.get_state_from_memory(oh_o_next)

        # reweight states using memory
        model.reweight_state(hd_state_pred_mem)

        state_pred_bind = weights                                       # eq (24)  u^{hat}_{t+1}
        state_pred_mem = model.compute_weights(hd_state_pred_mem)       # eq (25)  u^tilde}_{t+1}

        # weight computation, (32), w_hat_k are columns of w_hat
        w_tilde = model.compute_weights(model.M.T)                        # assume one-hot encoding of x_t

        if tt == 0 and ttt == 0: 
            print("oh_o_next_pred", oh_o_next_pred)
            print("Predicted state from binding\n", model.get_obs_score_from_memory(model.state))
            print("Predicted state from memory \n", model.get_obs_score_from_memory(hd_state_pred_mem))
            print("w^hat:   ", w_hat)
            print("w^tilde: ", w_tilde)

        # update rule, eq (31-34)
        self.__prep_update(w_tilde, w_hat, oh_a)                       # prepare for update, eq (31-34)

        # obsevation update rule, eq (31, 32)
        dQo = self.__update_Q_o(oh_o_next_pred, oh_o_next)
        dVo = self.__update_V_o(oh_o_next_pred, oh_o_next)

        # state update rule, eq (33, 34)
        dQs = self.__update_Q_s(state_pred_bind, state_pred_mem)
        dVs = self.__update_V_s(state_pred_bind, state_pred_mem)

        # Memory updates: memorize observation at time t+1; eq (20)
        model.update_memory(model.state, oh_o_next)                                          

        if normalize: 
            model.normalize_action() # normalize action

        # increment time 
        model.inc_time()

        loss = criterion(model.get_obs_score_from_memory(hd_s_pred_bind), o_next)

        return dQo + dQs, dVo + dVs, loss
    
    def __update_Q_o(self, oh_o_next_pred, oh_o_next_target):

        # TODO scaling factor for update (29)
        eta = self.lr_Q_o * self.alpha * self.beta_obs * self.lr_all
        
        update_weight = eta * torch.einsum('k, kmn', oh_o_next_target - oh_o_next_pred, self.Z_q)

        return update_weight        

    def __update_V_o(self, oh_o_next_pred, oh_o_next_target):

        # TODO scaling factor for update (30)
        eta = self.lr_V_o * self.alpha * self.beta_obs * self.lr_all

        update_weight = eta * torch.einsum('k, kmn', oh_o_next_target - oh_o_next_pred, self.Z_v)

        return update_weight 
        
    def __update_Q_s(self, state_pred_bind, state_pred_mem):

        eta = self.lr_Q_s * self.alpha * self.beta_state * self.lr_all

        update_weight = eta * torch.einsum('k, kmn', state_pred_mem - state_pred_bind, self.W_q)

        return update_weight

    def __update_V_s(self, state_pred_bind, state_pred_mem):

        eta = self.lr_V_s * self.alpha * self.beta_state * self.lr_all

        update_weight = eta * torch.einsum('k, kmn', state_pred_mem - state_pred_bind, self.W_v)
        
        return update_weight