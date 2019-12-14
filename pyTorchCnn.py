
'''
Written by: Rojan Shrestha PhD
Tue Nov 19 07:35:16 2019
'''

import torch
import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets
# import matplotlib.pylab as plt
# import numpy as np
# import pandas as pd

class MassCNN(nn.Module):

  def __init__(self, input, hidden, output, dropout=0.00):
    super(MassCNN,self).__init__(dropout=0.00)

    #first layers
    self.conv_L1        = nn.Conv1d(l_in=1, l_out=, kernel_size=1)
    self.maxpool1       = nn.MaxPool2d(kernel_size=2 )
    self.dropout_L1     = nn.Dropout(dropout)

    # Dense layer, no flattening is required 
    self.fc1 = nn.Linear(, 2)

  def forward(self, aX_tr):
    #first convolutional layers
    aX_tr = self.conv_L1(aX_tr)
    aX_tr = torch.relu(aX_tr)

    # get mean ... 
    # https://discuss.pytorch.org/t/how-to-implement-keras-layers-core-lambda-in-pytorch/5903
    # def select_top(x, k):
    X = torch.mean(T.sort(X, axis=1)[:, -k:, :], axis=1)
    aX_tr = torch.softmax(X)

    ### 
    pooled = Dropout(p=dropout)(pooled)
    #max pooling
    X = self.maxpool1(X)
    #flatten output
    X = X.view(X.size(0),-1)
    #fully connected layer
    X = self.fc1(X)
    return X

class ModelFit:

  def __init__(self, d_data, n_batch_size=100, b_shuffle=True):
    self._best_solution = {}
    af_Tr = DataLoader(d_trains, batch_size=n_batch_size, shuffle=b_shuffle)
    af_Va = DataLoader(d_valids, batch_size=n_batch_size, shuffle=b_shuffle)

  def generate_synthetic_data(self):
    """
       generate synthetic data to test the model

       params:

       returns: (aX_tr_sy, ay_tr_sy) - training pair - X and ground truth  
                (aX_va_sy, ay_va_sy) - validation pair - X and ground truth  
                (aX_te_sy, ay_te_sy) - testing pair - X and ground truth  
    """
    aX_tr_sy = np.random.random_sample(500000).reshape(1000, 50, 10)
    ay_tr_sy = np.random.randint(0, 2, 1000) 

    aX_va_sy = np.random.random_sample(50000).reshape(100, 50, 10)
    ay_va_sy = np.random.randint(0, 2, 100) 

    aX_te_sy = np.random.random_sample(50000).reshape(100, 50, 10)
    ay_te_sy = np.random.randint(0, 2, 100) 
    return (aX_tr_sy, ay_tr_sy), (aX_va_sy, aX_va_sy), (aX_te_sy, ay_te_sy)


  def test_hyper_parameters(self, n_trails=10, n_top=3):
 
    f_ts_lrs        = []
    n_ts_filters    = []
    f_ts_va         = []
    for n_train in range(n_trails): 
      t_lrs     = np.random() # get pytorch 
      n_filter  = np.random() # different filter 
      f_ts_lrs.append(t_lrs)
      n_ts_filters.append(n_ts_filters)
  
      f_ac_va = self.train(n_epochs=10, aX_tr, ay_tr, lr, n_filter)
      f_ts_va.append(f_ac_va)

    # get best result
    ar_idx = np.argsort(f_ts_va)[:n_top]
    self._best_solution['filter'] = 
    self._best_solution['lr'] = 
    self._best_solution['best_nets'] = 
    self._best_solution['best_accuracies'] = 

  def check_checkpoints(self, dir_name="checkpoint_models"):
    s_work_dir = os.path.join(os.getcwd(), dir_name)
    try:
      os.makedirs(s_work_dir)
    except FileExistsError:
      pass
    return s_work_dir 
    
      
      

  def train(self, n_epochs=10, aX_tr, ay_tr, lr=f_lr, filter=n_filter):
    """
    Train the model with Monte Carlo sampling in which hyperparameters
    are randomly generated. Multiple simulations are run to obtain 
    statistically robust models.

    params:

    returns: # of best solutions (models)
    """

    o_mass_model = MassCNN(50, filter, 3, 5)

    # Random data
    d_trains = SyntheticData(750, 5, 50, 3)
    d_valids = SyntheticData(750, 5, 50, 3)

    # optimization
    o_optimizer = torch.optim.SGD(o_mass_model.parameters(), lr=0.0001)
    o_cost      = torch.nn.CrossEntropyLoss()
  
    af_Tr = DataLoader(d_trains, batch_size=100, shuffle=True)
    af_Va = DataLoader(d_valids, batch_size=10, shuffle=True)

    t_solutions = []
    t_cost_per_epoch = []
    for epoch in range(n_epochs):
      m_models = dict()
      fgr_cost = 0
      for af_X, af_y in af_Tr:  # iterative over batch of data
        o_optimizer.zero_grad() ## Back propagation the loss
        af_y_pr  = o_mass_model(af_X.float())
        o_loss   = o_cost(af_y_pr, af_y)
        o_loss.backward()
        o_optimizer.step() # back propagate
        fgr_cost += o_loss.data
      t_cost_per_epoch.append(fgr_cost)
    
      #print(o_mass_model.state_dict().keys())
      #print("Epochs in fun ", epoch, id(o_mass_model))
      # Where is validation loss?
      o_mass_model.eval()
      f_correct = 0
      for af_x_va, af_y_va in af_Va:
        z  = o_mass_model(af_x_va.float()).data
        _, yhat = torch.max(z.data, 1)
        f_correct += (yhat == af_y_va).sum().item()
      f_accuracy = "%0.3f" %(f_correct / 1000) # fixed this with sample size
    
      m_models["epoch"] = epoch 
      m_models["iteration"] = n_iter
      m_models["val_acc"] = f_accuracy
      m_models["model_state"] =  o_mass_model.state_dict()
      m_models["optimizer_state"] =  o_optimizer.state_dict()
      m_models["loss"] =  o_cost.state_dict()
      m_models["model"] = o_mass_model
      t_solutions.append(m_models.copy())

      if b_checkpoint:
        s_fname      = "%s_%s" %(epoch, n_iter)
        s_model_path = os.path.join(self.check_checkpoints(), s_fname)
        torch.save(m_models, s_model_path)
      del m_models
    return t_solutions

  def get_best_models(t_models):
    """
    Select the best solutions from the pool
    
    params: 
      t_models: list of models. Each model is dictionary containing parameters
                and hyperparameters.
    return: 
      list of best models
    """
    n_best_sol      = 5
    t_best_val_accs = []
    t_best_models   = []
      
    for o_model in t_models: 
      b_update  = True
      f_val_acc = float(o_model["val_acc"])
      if len(t_best_val_accs) >= n_best_sol:
        print(t_best_val_accs)
        f_lowest_acc = np.min(t_best_val_accs)
        if f_val_acc > f_lowest_acc:
          i_cur_idx = t_best_val_accs.index(f_lowest_acc)
          t_best_val_accs.pop(i_cur_idx)
          t_best_models.pop(i_cur_idx)
        else:
          b_update = False
      if b_update: 
        t_best_val_accs.append(f_val_acc)
        t_best_models.append(o_model)
    return t_best_models

  def test_model(d_tests, t_best_sols):
    """
    Test performance of model on independent data. Since multiple models were generated, 
    these models were tested aganist an independent data and their average performance 
    will be returned as final result.
    
    params:
      d_tests: data for independent test
      t_best_sols: list of best solutions
    """
    t_test_accs = []
    for o_best_sol in t_best_sols:
      o_best_model = o_best_sol["model"]
      o_best_model.eval()
      f_correct = 0
      n_test_count = len(d_tests)
      for d_test_X, d_target_Y in d_tests:
        z  = o_best_model(d_test_X.float()).data # remove float
        _, yhat = torch.max(z.data, 1)
        f_correct += (yhat == d_target_Y).sum().item()
      f_accuracy = "%0.3f" %(f_correct / n_test_count) # fixed this with sample size
      t_test_accs.append(float(f_accuracy))
    print("Coding: ", t_test_accs)
    return np.mean(t_test_accs)



     
