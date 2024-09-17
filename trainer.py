import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm.notebook import tqdm
from copy import deepcopy

import wandb

from model import sim, POCML

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
        self.lr_all = lr_all
        self.normalize = normalize
        self.reset_every = reset_every
        self.update_state_given_obs = update_state_given_obs

        self.step_ct = 0                                        # step count
        self.traj_ct = 0                                        # trajectory count
        self.epoch_ct = 0                                       # epoch count

        self.debug = debug
        self.log = log

    # # Create tensor reused in the update rule (30 - 33)
    # # the tensor U is of shape n_s * n_s * n where U[i,j,k] = \omega_{i,j,k} K_{i，j} (s^{hat}_j - s_i) 
    def __prep_update(self, oh_o_next, oh_a):
        alpha = self.model.random_feature_map.alpha

        model: POCML = self.model
        Q = model.Q

        v_t = model.V @ oh_a                                # v_t []
        s_hat = (Q.T) + v_t[None, :]                        # s_hat_j = (s_j + v_t)
        self.diff_s = (Q.T)[:, None, :] - s_hat[None, :, :]      # generate n_s * n_s * n tensor [s_i - s_hat_j]_i,j

        phi_Q = model.get_state_kernel()
        phi_s = model.state # \phi(\hat{s}_t)
        phi_v = model.random_feature_map(v_t).unsqueeze(dim=1)

        sims_pairwise = sim(phi_Q, phi_Q * phi_v)
        sims_s_hat = sim(phi_Q, phi_s)
        # diff_s_squared = torch.einsum("ijk,ijk->ij", self.diff_s, self.diff_s)
        # sims_pairwise = torch.exp(-alpha * diff_s_squared)
        # sims_s_hat = sims_pairwise @ model.u
        p = model.get_obs_from_memory(torch.eye(model.n_states)) @ oh_o_next
        self.p = p
        self.sims_s_hat = sims_s_hat

        r1 = torch.einsum("i,ij->ij", p, sims_pairwise) / (p @ sims_s_hat)
        r2 = sims_pairwise / sims_s_hat.sum()
        self.gamma = r1 - r2

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
            if mean_loss == np.nan:
                print("mean loss is nan")
                break
            if mean_loss < best_loss:
                best_loss, best_model = mean_loss, deepcopy(self.model)

            self.epoch_ct += 1
            if self.log:
                wandb.log({"train/mloss_p_epoch": mean_loss,
                            "train/epoch_ct": self.epoch_ct})
        
        return np.array(loss_record).reshape(epochs,-1), best_model
    
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

                oh_o_first = F.one_hot(trajectory[0,0,0], num_classes=model.n_obs).to(torch.float32)

                # TODO reset graph instance
                if tt % self.reset_every == 0:
                    #model.init_memory(model.M.detach() / 10)
                    model.init_memory()
                
                # train model
                #model.init_state(obs = oh_o_first) #  treat the first observation as the special case. 
                model.init_state(state_idx=trajectory[0,0,4])
                model.update_memory(model.u, oh_o_first)        #  memorize the first observation

                phi_Q = model.get_state_kernel()
                if self.debug:
                    print("Current Trajectory", trajectory[0])
                    print("Action similarities\n", model.get_action_similarities())
                    print("State kernel similarities (want close to identitiy)\n", sim(phi_Q, phi_Q))
                
                # o_pre  is the observation at time t
                for ttt, (o_pre, a, o_next, _, _) in enumerate(trajectory[0].to(device)):
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
        
        # Compute and apply update rule
        self.__prep_update(oh_o_next, oh_a)
        self.__update_Q(oh_u_pre)
        self.__update_V(oh_a, oh_u_pre)

        # Clean up state \phi(\hat{s}_{t+1})
        if self.update_state_given_obs:
            model.update_state_given_obs(oh_o_next) # set u_{t+1} to p(u_{t+1} | s_{t+1}, x_{t+1} )
        model.clean_up()

        # Update memory
        model.update_memory(oh_u_next, oh_o_next)

        if normalize: 
            model.normalize_action() # normalize action

        model.inc_time()

        loss = -torch.log(oh_o_next_pred[o_next]) # cross entropy

        if self.debug: 
            print("Time", model.t)
            print("o_pre, o_next", o_pre, o_next)
            print("Expected previous state", oh_u_pre)
            print("Expected next state", oh_u_next)
            if self.update_state_given_obs:
                print("Expected next state given obs", model.u)
            print("Predicted obs from action", oh_o_next_pred)
            print("gamma", self.gamma)
            print("p", self.p)
            print("sims_s_hat", self.sims_s_hat)

        return loss
    
    def __update_Q(self, oh_u_pre):
        eta = self.lr_Q * self.alpha * self.lr_all
        u = torch.eye(self.model.n_states).to(self.device)
        dQ = eta * torch.einsum("ij,j,ijk,il->kl", self.gamma, oh_u_pre, -self.diff_s, u)
        self.model.Q += dQ
        return dQ

    def __update_V(self, oh_a, oh_u_pre):
        eta = self.lr_V * self.alpha * self.lr_all
        dV = eta * torch.einsum("ij,j,ijk,l->kl", self.gamma, oh_u_pre, self.diff_s, oh_a)
        self.model.V += dV
        return dV
    
class BenchmarkTrainer:
    def __init__(self,
                 model,
                 train_loader,
                 optimizer=None,
                 criterion=None,
                 val_loader=None,
                 device=None,
                 ):

        #super.__init__(model, train_loader, norm=False, optimizer=None, criterion=None, val_loader=None, device=None)
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_loader = val_loader
        self.device = device

        self.dataset = self.__preprocess_dataset(self.train_loader)

    def train(self, epochs:int = 10) -> list:
        loss_record = []

        for epoch in tqdm(range(epochs), desc="Epochs"):
            last_loss = self.train_epoch() # Concatenate the list of losses
            loss_record += last_loss # Concatenate the list of losses
        
        return np.array(loss_record).reshape(epochs,-1), self.model
    
    def __preprocess_dataset(self, dataloader):
        model: POCML = self.model
        dataset = []
        for traj in dataloader:
            x = traj[0][:, :2]
            y = traj[0][:, 3]
            init_state = traj[0][0, 4]
            new_x = torch.stack([
                torch.cat([
                    F.one_hot(x[i, 0], num_classes=model.n_obs),
                    F.one_hot(x[i, 1], num_classes=model.n_actions)
                ]) for i in range(x.shape[0])
            ])
            new_y = torch.stack([
                F.one_hot(y[i], num_classes=model.n_obs) for i in range(y.shape[0])
            ])
            z = F.one_hot(init_state, num_classes=model.n_states)
            dataset.append((new_x, new_y, z))

    def train_epoch(self) -> list:
        model: POCML = self.model
        loss_record = []     
        
        for x, y, init_state in self.dataset:
            y_pred = model(x, init_state)
            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_record.append(loss.item())

        return loss_record