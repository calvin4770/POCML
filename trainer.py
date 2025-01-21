import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from copy import deepcopy

import wandb

from model import sim, POCML
from dataloader import preprocess_dataset

class CMLTrainer:
    def __init__(self, model, train_loader, norm=False, optimizer=None, criterion=None, val_loader=None, device=None, debug =False):
        
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_loader = val_loader
        self.device = device
        self.norm = norm
        self.debug = debug

    def train(self, epochs = 10):

        best_model = None
        best_loss = 1e10
        loss_record = []
        for _ in tqdm(range(epochs), desc="Epochs"):
            losses = self.train_epoch()
            mean_loss = np.mean(losses)
            loss_record += losses # Concatenate the list of losses
            if mean_loss < best_loss:
                best_loss, best_model = mean_loss, deepcopy(self.model)

        return np.array(loss_record).reshape(epochs,-1), best_model

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
                 lr_Q = 1.,
                 lr_V = 0.1,
                 lr_all = 0.32,
                 lr_M = 0.1,
                 reg_Q = 0,
                 reg_V = 0,
                 reg_M = 0.1,
                 max_iter_M=50, eps_M=1e-3,
                 reset_every = 1, # reset every N trajectories,
                 update_state_given_obs=False,
                 normalize = False,
                 debug = False,
                 log = False,
                 ):

        #super.__init__(model, train_loader, norm=False, optimizer=None, criterion=None, val_loader=None, device=None)
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_loader = val_loader
        self.device = device
        self.norm = norm

        # POCML model param for training
        self.alpha = model.random_feature_map.alpha             # (inverse) lengscale of the RBF
        self.lr_Q = lr_Q
        self.lr_V = lr_V
        self.lr_M = lr_M
        self.lr_all = lr_all
        self.reg_Q = reg_Q
        self.reg_V = reg_V
        self.reg_M = reg_M
        self.normalize = normalize
        self.reset_every = reset_every
        self.update_state_given_obs = update_state_given_obs
        self.max_iter_M = max_iter_M
        self.eps_M = eps_M

        self.step_ct = 0                                        # step count
        self.traj_ct = 0                                        # trajectory count
        self.epoch_ct = 0                                       # epoch count

        self.debug = debug
        self.log = log

    # # Create tensor reused in the update rule (30 - 33)
    # # the tensor U is of shape n_s * n_s * n where U[i,j,k] = \omega_{i,j,k} K_{i，j} (s^{hat}_j - s_i) 
    def __prep_update(self, oh_o_next, oh_a):
        relu = lambda x: torch.maximum(x, torch.ones_like(x) * 1e-6)
        model: POCML = self.model
        Q = model.Q

        v_t = torch.einsum("da,ba->bd", model.V, oh_a)
        s_hat = Q.unsqueeze(0) + v_t.unsqueeze(-1)          # s_hat_j = (s_j + v_t) [B, state_dim, n_states]
        self.diff_s = model.Q.unsqueeze(0)[:, :, :, None] - s_hat[:, :, None, :] # [B, D, n_s, n_s]
        
        phi_Q = model.get_state_kernel()
        phi_s = model.state # \phi(\hat{s}_t)
        phi_v = model.random_feature_map(v_t)

        phi_Q2 = phi_Q.unsqueeze(0) * phi_v.unsqueeze(-1) # [B, D, n_s]
        sims_pairwise = relu(torch.einsum("ds,bdt->bst", phi_Q, phi_Q2.conj()).real / phi_Q.shape[0]) # [B, n_s, n_s]
        sims_s_hat = sim(phi_Q, phi_s.T).T # [B, n_s]
        self.sims_s_hat = sims_s_hat

        r1 = sims_pairwise / (sims_s_hat.sum(dim=1))[:, None, None]
        r2 = torch.einsum("bj,bjk,bj->bjk", model.u, sims_pairwise, 1/sims_s_hat)
        self.gamma = r2 - r1

    def train(self, epochs:int = 10) -> list:

        best_model = None
        best_loss = 1e10
        loss_record = []

        for epoch in tqdm(range(epochs), desc="Epochs"):
            if self.debug:
                print(f"===========Epoch {epoch}===========")
            last_loss = self.train_epoch() # Concatenate the list of losses
            loss_record += last_loss # Concatenate the list of losses
            mean_loss = np.mean(last_loss)
            if np.isnan(mean_loss):
                print("mean loss is nan")
                break
            if mean_loss < best_loss:
                best_loss, best_model = mean_loss, deepcopy(self.model)

            self.epoch_ct += 1
            if self.log:
                wandb.log({"train/mloss_p_epoch": mean_loss,
                            "train/epoch_ct": self.epoch_ct})
        
        return np.array(loss_record).reshape(epoch+1,-1), best_model
    
    ## Naming convention
    #    hd_, sa_, oh_/*: respective objects in HD space, state-action space, and ``one_hot" space
    #    a, o, s: action, observation, state (note that "u" in the paper is replaced with s; hd_s and as_s is used to disambiguate the domain)
    #    _bind, _mem: state inferences are made from binding (eq (18,19), denoted \hat) vs from memory (eq 22, \tilde)
    def train_epoch(self) -> list:

        with torch.no_grad():

            model: POCML = self.model
            device = self.device
            normalize = self.normalize
            loss_record = []

            # memory transfer option/reset rate + for decay design
            model.init_memory()                    # reset model's memory for new epochs        
            
            for tt, trajectory in enumerate(self.train_loader):

                model.init_time()

                oh_o_first = F.one_hot(trajectory[:,0,0], num_classes=model.n_obs).to(torch.float32)

                # reset memory every N trajectories
                if tt % self.reset_every == 0:
                    #model.init_memory(model.M.detach() / 10)
                    model.init_memory()
                
                # train model
                #model.init_state(obs = oh_o_first) #  treat the first observation as the special case. 
                model.init_state(state_idx=trajectory[:,0,4])
                model.update_memory(model.u, oh_o_first, lr=self.lr_M, reg_M=self.reg_M, max_iter=self.max_iter_M, eps=self.eps_M)        #  memorize the first observation

                phi_Q = model.get_state_kernel()
                if self.debug:
                    print("Current Trajectory", trajectory[:])
                    print("Action similarities\n", model.get_action_similarities())
                    print("State kernel similarities (want close to identitiy)\n", sim(phi_Q, phi_Q))
                
                # o_pre  is the observation at time t
                for t in range(trajectory.shape[1]):
                    o_pre = trajectory[:, t, 0]
                    a = trajectory[:, t, 1]
                    o_next = trajectory[:, t, 2]

                    loss = self.__one_time_step(model, o_pre, a, o_next, normalize=normalize)
                    loss_record.append(loss.cpu().item())
                    
                    self.step_ct += 1
                    if self.log:
                        wandb.log({"train/loss": loss,
                                    "train/step_ct": self.step_ct,})

                self.traj_ct += 1
                if self.log:
                    #print("Debug", loss, len(trajectory[0]))
                    wandb.log({"train/mloss_p_traj": sum(loss_record[-len(trajectory[0]):])/len(trajectory[0]),
                               "train/traj_ct": self.traj_ct})

        return loss_record
    
    def __one_time_step(self, model: POCML, o_pre, a, o_next, normalize=False):
        oh_o_next = F.one_hot(o_next, num_classes=model.n_obs).to(torch.float32)  # one-hot encoding of the first observation
        oh_a = F.one_hot(a, num_classes=model.n_actions).to(torch.float32)     # one-hot encoding of the first observation
        
        oh_u_pre = model.u # u_t
        hd_s_pred_bind_precleanup = model.update_state(oh_a) # update state by binding action
        oh_u_next = model.get_expected_state(hd_s_pred_bind_precleanup) # get p(u | \phi(\hat{s}_{t+1}'))
        oh_o_next_pred = model.get_obs_from_memory(oh_u_next) # predict observation with updated state
        model.update_state_given_obs(oh_o_next)

        # Compute and apply update rule
        self.__prep_update(oh_o_next, oh_a)
        self.__update_Q(oh_u_pre)
        self.__update_V(oh_a, oh_u_pre)

        # Clean up state \phi(\hat{s}_{t+1})
        model.clean_up()

        # Update memory
        model.update_memory(oh_u_next, oh_o_next, lr=self.lr_M, reg_M=self.reg_M, max_iter=self.max_iter_M, eps=self.eps_M)

        if normalize: 
            model.normalize_action() # normalize action

        model.inc_time()

        loss = -torch.log(torch.einsum("bi,bi->b", oh_o_next_pred, oh_o_next)).mean() # cross entropy

        if self.debug: 
            print("Time", model.t)
            print("o_pre, o_next", o_pre, o_next)
            print("Expected previous state", oh_u_pre)
            print("Expected next state", oh_u_next)
            if self.update_state_given_obs:
                print("Expected next state given obs", model.u)
            print("Predicted obs from action", oh_o_next_pred)
            print("Gamma", self.gamma)

        return loss
    
    def __update_Q(self, oh_u_pre):
        eta = self.lr_Q * self.alpha * self.lr_all
        u = torch.eye(self.model.n_states).to(self.device)
        dQ = eta * torch.einsum("bij,bj,bkij,il->kl", self.gamma, oh_u_pre, -self.diff_s, u) / self.model.batch_size
        reg = -self.model.Q # L2 reg
        # TODO orthogonality reg on phi(Q) OR pairwise state differences large (exp scale)
        self.model.Q += dQ + self.reg_Q * reg
        return dQ

    def __update_V(self, oh_a, oh_u_pre):
        eta = self.lr_V * self.alpha * self.lr_all
        dV = eta * torch.einsum("bij,bj,bkij,bl->kl", self.gamma, oh_u_pre, self.diff_s, oh_a) / self.model.batch_size
        reg = -self.model.V
        self.model.V += dV + self.reg_V * reg
        return dV
    
class BenchmarkTrainer:
    def __init__(self,
                 model,
                 train_loader,
                 optimizer=None,
                 criterion=None,
                 test_loader=None,
                 device=None,
                 include_init_state_info=True,
                 reset_every=1,
                 log = False,
                 ):

        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.test_loader = test_loader
        self.device = device
        self.reset_every = reset_every
        self.include_init_state_info = include_init_state_info
        self.log = log

        self.train_dataset = preprocess_dataset(model, train_loader, include_init_state_info=include_init_state_info)
        self.test_dataset = preprocess_dataset(model, test_loader, include_init_state_info=include_init_state_info)

        traj_length = self.train_dataset[0][1].shape[0]
        start = 1
        idxs = []
        for _ in range(self.reset_every):
            idxs += list(range(start, start + traj_length))
            start += traj_length + 1
        self.idxs = idxs

    def train(self, epochs:int = 10) -> list:
        loss_record = []
        
        for epoch in tqdm(range(epochs), desc="Epochs"):
            last_loss = self.train_epoch() # Concatenate the list of losses
            loss_record += last_loss # Concatenate the list of losses
        
        return np.array(loss_record).reshape(epochs,-1), self.model

    def train_epoch(self) -> list:
        model = self.model
        loss_record = []
        
        for i, (x, y) in enumerate(self.train_dataset):
            if i % self.reset_every == 0:
                if i > 0:
                    model.reset_state()
                    x_agg = torch.cat(x_agg, dim=0)
                    y_agg = torch.cat(y_agg, dim=0)
                    y_pred_agg = model(x_agg)
                    if self.include_init_state_info:
                        y_pred_agg = y_pred_agg[self.idxs, :]
                    loss = self.criterion(y_pred_agg, y_agg)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                x_agg, y_agg = [], []
            x_agg.append(x)
            y_agg.append(y)

            with torch.no_grad():
                y_pred = model(x)
                if self.include_init_state_info:
                    y_pred = y_pred[1:, :]
                loss = self.criterion(y_pred, y)
            loss_record.append(loss.item())

            if self.log:
                wandb.log({"train/loss": loss.item()})

        return loss_record