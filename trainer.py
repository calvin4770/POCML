import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

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

    # Sample validation code
    # def validate_epoch(self):
    #     self.model.eval()
    #     total_loss = 0
    #     with torch.no_grad():
    #         for data, labels in self.val_loader:
    #             outputs = self.model(data)
    #             loss = self.criterion(outputs, labels)
    #             total_loss += loss.item()
    #     return total_loss / len(self.val_loader)

    # Sample train
    # def train(self, num_epochs):
    #     for epoch in range(num_epochs):
    #         train_loss = self.train_epoch()
    #         print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
    #         if self.val_loader:
    #             val_loss = self.validate_epoch()
    #             print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')




class POCMLTrainer(CMLTrainer):
    def __init__(self, model, train_loader, norm=False, optimizer=None, criterion=None, val_loader=None, device=None):

        #super.__init__(model, train_loader, norm=False, optimizer=None, criterion=None, val_loader=None, device=None)
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_loader = val_loader
        self.device = device
        self.norm = norm

        # POCML model param for training
        self.beta = model.beta                                  # state prediction temperature, eq(21)
        self.alpha = model.random_feature_map.alpha             # (inverse) lengscale 
        # TODO D, depending on normalization ~~~ learning rate
        self.lr_Q_o = 1
        self.lr_V_o = 0.1
        self.lr_Q_s = 0.0
        self.lr_V_s = 0.0
        self.lr_all = 0.032

    # Create tensor reused in the update rule (30 - 33)
    def __prep_update(self, w_tilde, w_hat, oh_a):
        
        model = self.model 
        Q = model.Q

        omega = torch.outer(w_tilde, w_hat)                 # omega_ij = w_tilde_i * w_hat_j

        v_t = model.V @ oh_a
        s_hat = (Q.T) + v_t[None, :]                        # s_hat_j = (s_j + v_t)
        diff_s = (Q.T)[:, None, :] - s_hat[None, :, :]      # generate n_s * n_s * n tensor [s_i - s_hat_j]_i,j
        
        diff_s_squared_norm = torch.sum(diff_s ** 2, dim=-1)
        K = torch.exp(-self.alpha * diff_s_squared_norm)

        self.update_tensor = (omega * K)[:, : ,None] * diff_s
        #self.update_tensor = torch.einsum('ij,ijm->ijm', omega * K, diff_s)    # Alternatively


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
            norm = self.norm
            loss_record = []

            # memory transfer option/reset rate + for decay design
            model.init_memory()                    # reset model's memory for new epochs        
            
            for trajectory in self.train_loader:

                model.init_time()
                
                #if model.reset_per_trajectory:             # memory have to option to reset per trajectory
                #    model.init_memory()
                                                            # TODO memory should have the option to reset per graph instance

                oh_o_first = F.one_hot(trajectory[0,0,0], num_classes=model.n_obs).to(torch.float32)
                
                model.init_state(obs = oh_o_first)                  #  treat the first observation as the spacial case. 
                model.update_memory(model.state, oh_o_first)        #  memorize the first observation

                # print(model.state)
                # print(model.M.T)
                # print(model.state.norm(p=2))
                # print(model.M.T.norm(dim=0, p = 2))
                print("initial state:", trajectory[0,0,0])
                print("Print initial score", model.get_obs_score_from_memory(model.state))
                print("Obs similarity", (model.M.T @ model.M.conj()).real)
                #print("State difference", model.Q.T[None, :, :] model.Q)
                print("State difference", (model.Q[:, :, None] - model.Q[:, None, :]).norm(p=2, dim=0))

                hd_s_pred_bind_precleanup_t = model.state               # initialize state prediction from binding at time t, used in equation (29)
                                                                        # at t = 0, this is initialized as initial state       

                # o_pre  is the observation at time t
                for o_pre, a, o_next in trajectory[0].to(device):

                    print("Time", model.t)
                    
                    oh_o_pre = F.one_hot(trajectory[0,0,0], num_classes=model.n_obs).to(torch.float32)  # one-hot encoding of the first observation
                    oh_o_next = F.one_hot(trajectory[0,0,0], num_classes=model.n_obs).to(torch.float32)  # one-hot encoding of the first observation
                    oh_a = F.one_hot(trajectory[0,0,0], num_classes=model.n_actions).to(torch.float32)     # one-hot encoding of the first observation
                    
                    # hd_s_pred_bind = model.infer_hd_state_from_binding(o_pre, oh_a)   # infer state via binding at time t+1, s^{hat}^{prime}_{t+1}, eq (18)
                    hd_s_pred_bind_precleanup = model.update_state(oh_a)

                    print("Print curent state", model.get_obs_score_from_memory(model.state))
                    
                    #hd_state_pred_mem = self.model.infer_hd_state_from_memory(oh_o_next) # infer state at time t+1 via memory, s^{tilde}_{t+1} eq (22)
                    hd_state_pred_mem = model.get_state_from_memory(oh_o_next)

                    # hd_s_pred_bind = self.model.cleanup_hd_state(hd_s_pred_bind)   # clean up state, s^{hat}_{t+1}, eq (19)
                    weights = model.clean_up(
                        hd_s_pred_bind_precleanup, 
                        state_from_memory=hd_state_pred_mem,
                        c = 1.0,
                    ) # coeff for eq (19); use for eq (24)
                    hd_s_pred_bind = model.state            

                    #o_next_pred = self.model_predict_obs(hd_s_pred_bind) # predict observation at time t+1, x^{hat}_{t+1} eq (21)
                    oh_o_next_pred = model.get_obs_from_memory(hd_s_pred_bind)
                    #print(model.get_obs_score_from_memory(hd_s_pred_bind))

                    state_pred_bind = weights                                       # eq (24)  u^{hat}_{t+1}
                    state_pred_mem = model.compute_weights(hd_state_pred_mem)  # eq (25)

                    # weight computation, (28) (29)
                    w_hat = model.compute_weights(hd_s_pred_bind_precleanup_t)                  # s^{hat}_t^{prime} is not computed in this iteration eq (29)
                                                                                                # TODO option to use s^{hat}_t; 
                    w_tilde = state_pred_mem                                                    # eq (30) = (25)

                    # update rule, eq (31-34)

                    update = self.__prep_update(w_hat, w_tilde, oh_a)                       # prepare for update, eq (31-34)

                    # obsevation update rule, eq (31, 32)
                    print("dQ: ", self.__update_Q_o(oh_o_next_pred, oh_o_next))                             # update observation for time t+1, x_{t+1} eq (31, 32)  
                    print("dV:" , self.__update_V_o(oh_o_next_pred, oh_o_next, oh_a))                       #

                    # state update rule, eq (33, 34) TODO
                    self.__update_Q_s(state_pred_bind)                                      # update observation for time t+1, x_{t+1} eq (32, 33)
                    self.__update_V_s(state_pred_bind, oh_a)                                # 

                    # self.model_update_memory(o_next) # memorize observation at time t+1 for eq (20)
                    model.update_memory(hd_s_pred_bind, oh_o_next)
                    # keep the state prediction from binding at time t+1 to used in the next iteration for equation (29)
                    hd_s_pred_bind_precleanup_t = hd_s_pred_bind_precleanup

                    model.inc_time()                                  # increment time 

                    loss = nn.CrossEntropyLoss()(model.get_obs_score_from_memory(hd_s_pred_bind), o_next)
                    # loss = nn.CrossEntropyLoss()(oh_o_next_pred, oh_o_next)
                    # loss = nn.MSELoss()(oh_o_next_pred, oh_o_next)
                    # loss_hidden = nn.CrossEntropyLoss()(hd_s_pred_bind, hd_state_pred_mem)
                    loss_record.append(loss.cpu().item())

        return loss_record

            
    
    def __update_Q_o(self, oh_o_next_pred, oh_o_next_target):

        eta = self.lr_Q_o * self.alpha * self.beta * self.lr_all
        
        u = torch.eye(self.model.n_states).to(self.device)        # TODO double check if this can be optimized
        update_weight = eta * (1 - torch.dot(oh_o_next_pred, oh_o_next_target)) * \
                        torch.einsum('ijk,jl->kl', self.update_tensor, u)
        self.model.Q += update_weight
        return update_weight
        

    def __update_V_o(self, oh_o_next_pred, oh_o_next_target, oh_a):

        eta = self.lr_V_o * self.alpha * self.beta * self.lr_all
        update_weight = eta * (1 - torch.dot(oh_o_next_pred, oh_o_next_target)) * \
                        torch.einsum('ijk,l->kl', self.update_tensor, oh_a)
        self.model.V += update_weight
        return update_weight 
        
    def __update_Q_s(self, state_pred_bind):

        eta = self.lr_Q_s * self.alpha * self.beta * self.lr_all
       
        u = torch.eye(self.model.n_states).to(self.device)        # TODO double check if this can be optimized
        update_weight = eta * \
                        torch.einsum('i,ijk,jl->kl', state_pred_bind, self.update_tensor, u)
        self.model.Q += update_weight
        return update_weight

    def __update_V_s(self, state_pred_bind, oh_a):

        eta = self.lr_V_s * self.alpha * self.beta * self.lr_all
        update_weight = eta * \
                        torch.einsum('i,ijk,l->kl', state_pred_bind, self.update_tensor, oh_a)
        self.model.V += update_weight
        return update_weight



