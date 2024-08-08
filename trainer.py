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

        model = self.model 
        device = self.device
        norm = self.norm
        loss_record = []

        # POCML model param for training
        beta = model.beta # state prediction temperature, eq(21)
        # alpha / lengscale * beta ~~~ learning rate 
        # D, depending on normalization ~~~ leanring rate

        # memory transfer option/reset rate + for decay design
        self.model.init_memory()                    # reset model's memory for new trajectory          
        
        for trajectory in self.train_loader:

            t = 0                                       # time start at zero
            
            if model.reset_per_trajectory:         # memory have to option to reset per trajectory
                model.init_memory()
                                                        # TODO memory should have the option to reset per graph instance

            oh_o_first = F.one_hot(trajectory[0,0,0], num_classes=model.n_obs).to(torch.complex64)
            
            model.init_state(obs = oh_o_first)                 #  treat the first observation as the spacial case. 
            model.update_memory(self.model.state, oh_o_first)  # memorize the first observation

            hd_s_pred_bind_precleanup_t = model.state               # initialize state prediction from binding at time t, used in equation (28)
                                                                    # at t = 0, this is initialized as initial state       

            # o_pre  is the observation at time t
            for o_pre, a, o_next in trajectory[0].to(device):
                
                oh_o_pre = F.one_hot(trajectory[0,0,0], num_classes=model.n_obs).to(torch.complex64)  # one-hot encoding of the first observation
                oh_o_next = F.one_hot(trajectory[0,0,0], num_classes=model.n_obs).to(torch.complex64)  # one-hot encoding of the first observation
                oh_a = F.one_hot(trajectory[0,0,0], num_classes=model.n_actions).to(torch.complex64)  # one-hot encoding of the first observation

                # hd_s_pred_bind = model.infer_hd_state_from_binding(o_pre, oh_a)   # infer state via binding at time t+1, s^{hat}^{prime}_{t+1}, eq (18)
                hd_s_pred_bind_precleanup = model.update_state(oh_a)
                
                # hd_s_pred_bind = self.model.cleanup_hd_state(hd_s_pred_bind)   # clean up state, s^{hat}_{t+1}, eq (19)
                weights = model.cleanup(hd_s_pred_bind) # coeff for (19); use for (23)
                hd_s_pred_bind = model.state            

                #o_next_pred = self.model_predict_obs(hd_s_pred_bind) # predict observation at time t+1, x^{hat}_{t+1} eq (21)
                oh_o_next_pred = model.get_obs_from_memory(hd_s_pred_bind)

                #hd_state_pred_mem = self.model.infer_hd_state_from_memory(oh_o_next) # infer state at time t+1 via memory, s^{tilde}_{t+1} eq (22)
                hd_state_pred_mem = model.get_state_from_memory(self, oh_o_next)

                state_pred_bind = weights                                       # (23)  u^{hat}_{t+1}
                state_pred_mem = self.model.compute_weights(hd_state_pred_mem)  # (24) 

                # weight computation, (28) (29)
                w_hat = self.model.infer_state_from_hd_state(hd_s_pred_bind)            # s^{hat}_t^{prime} is not computed in this iteration eq (28)
                                                                                        # s^{hat}_t; 
                                                                                        # edge case: get weight from initial states (t=0)
                w_tilde = state_pred_mem                                                # eq (29) = (24)

                model.prep_update()                                                # prepare for update, eq (30-33)          
                # obsevation update rule, eq (30, 31)
                model.update_Q_o()                             # update observation for time t+1, x_{t+1} eq (30, 31)  
                model.update_V_o()                             #

                # state update rule, eq (32, 33)
                model.update_Q_s()                             # update observation for time t+1, x_{t+1} eq (32, 33)
                model.update_V_s()                             # 

                # self.model_update_memory(o_next) # memorize observation at time t+1 for eq (20)
                model.update_memory(hd_s_pred_bind, oh_o_next)

                hd_s_pred_bind_precleanup_t = hd_s_pred_bind_precleanup

                t += 1                                  # increment time 

            # collect loss; would likely have to be in the inner loop.
            loss = -1                                           # TODO (model) compute loss for the current time step   
            loss_record.append(loss.cpu().item())




