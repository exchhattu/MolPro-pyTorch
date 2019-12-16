'''
Written by: Rojan Shrestha PhD
Mon Nov 18 17:35:38 2019
'''

import sys, os, errno

import numpy as np

import flowio # pip install FlowIO

class FCdata:
  
  def __init__(self, path_to_dir, path_to_label_data, path_to_marker):
    """
     
    Params:
      st_path_to_file: 
      st_path_to_label: 
      st_path_to_marker: 
    """
    self._ma_data    = {}

    self._Xtrains   = []
    self._Xvalids   = []
    self._Xtest     = []
    self._Ytrains   = []
    self._Yvalids   = []
    self._Ytest     = []

    # get ...
    oj_idata = self.iData(path_to_dir, path_to_label_data, path_to_marker)
    print("coding: 23 ", oj_idata._ts_samples)
    print("coding: 12 ", oj_idata._ts_phenotypes) 
    # for st_path, st_label in oj_idata._ma_labels.items():
    #   print("Coding: ", st_path, st_label)
    #   ar_events, ts_channels = oj_idata.read_flowdata(st_path, 
    #                                                   markers = oj_idata._ts_markers, 
    #                                                   transform=None, 
    #                                                   auto_comp=False)
    #   self._ma_data[st_path] = ar_events


  def load_data(self): 
    """
      
    """
    in_num_sample   = len(self._ma_labels)
    in_train_sample = int(0.70*in_num_sample)
    in_valid_sample = int(0.15*in_num_sample)
    in_test_sample  = int(0.15*in_num_sample)

    ar_idx          = np.random.permutation(in_num_sample)
    ar_keys         = self._ma_labels.keys()

    ar_keys[ar_idx[:in_train_sample]]
    ar_idx[in_train_sample:in_train_sample+in_valid_sample]
    ar_idx[-in_test_sample:]
    self._Xtrains   = []
    self._Xvalids   = []
    self._Xtest     = []
    self._Ytrains   = []
    self._Yvalids   = []
    self._Ytest     = []

    # return ...

  def combine_samples(self, data_list, sample_id):
    """
      Aims: merge multiple samples together, which is identified by their sample id.
            index of data_list and sample_id should be synchronized.
      Params: 
        data_list - list of sample data
        sample_id - list of sample ids     
    """
    accum_x, accum_y = [], []
    for x, y in zip(data_list, sample_id):
      accum_x.append(x)
      accum_y.append(y * np.ones(x.shape[0], dtype=int))
    return np.vstack(accum_x), np.hstack(accum_y)

  def generate_subsets(self, X, pheno_map, ts_sample_ids, nsubsets=1000, 
                       ncell=200, per_sample=False, k_init=False):
    """
      Aims: generates the data ready for pytorch model. This data generation
            is very problem specific. Each patient has nsubsets data and 
            each contains ncell.

      Params:

    """
    S = dict()
    n_unique_sample = len(np.unique(ts_sample_ids))

    # create N subset samples for each patient. each subset contains 
    # N randomly selected cells  
    for n_sample_id in range(n_unique_sample):
      X_i            = X[np.where(ts_sample_ids == n_sample_id)]
      S[n_sample_id] = per_sample_subsets(X_i, nsubsets, ncell, k_init)
      # contains 3D data

    # interesting going here - onward data will not keep track of patient
    # information instead there will be phenotype. Since S.values() is
    # three dimensional array, patient specific data is not mingled with
    # others
    data_list, y_list = [], []
    for y_i, x_i in S.items(): # y_i: patient ids and x_i: their corresponding cells 
      data_list.append(x_i)
      y_list.append(pheno_map[y_i] * np.ones(x_i.shape[0], dtype=int))

    Xt = np.vstack(data_list)
    yt = np.hstack(y_list)
    Xt, yt = sku.shuffle(Xt, yt)
    return Xt, yt

  def per_sample_subsets(self, X, nsubsets, ncell_per_subset, k_init=False):
    """
      Aims: prepare the dimension ready to input the deep learning model

      Params:
        
    """
    nmark = X.shape[1]
    shape = (nsubsets, nmark, ncell_per_subset)
    Xres = np.zeros(shape)

    if not k_init:
      for i in range(nsubsets):
        X_i = random_subsample(X, ncell_per_subset)
        Xres[i] = X_i.T
    else:
      for i in range(nsubsets):
        X_i = random_subsample(X, 2000)
        X_i = kmeans_subsample(X_i, ncell_per_subset, random_state=i)
        Xres[i] = X_i.T
    return Xres


  class iData:
   
    def __init__(self, path_to_dir, path_to_label, path_to_marker, cofactor=5):

      self._ma_labels  = dict() 
      self._ts_markers = []

      self.read_labels(path_to_label)   # label either positive or neutral
      self.read_markers(path_to_marker) # marker of each cell

      self._ts_samples      = []
      self._ts_phenotypes   = []

      # read all files with suffix .fcs from the given directory.
      for fname, flabel in self._ma_labels.items():
        full_path   = os.path.join(path_to_dir, fname)
        ar_events, ts_channels  = self.read_flowdata(full_path, transform=None, auto_comp=False)

        ts_marker_idx = [ts_channels.index(name) for name in self._ts_markers]
        x = ar_events[:, ts_marker_idx]
        x = np.arcsinh(1./cofactor * x)
        self._ts_samples.append(x)
        self._ts_phenotypes.append(flabel)

    def read_labels(self, path_to_label):
      """
        Read the label of each mass cytometry file and store into dictionary

        Params:
          path_to_label: path to label file
      """

      if os.path.exists(path_to_label):
        with open(path_to_label, "r") as oj_path:
          ts_fcm_files = oj_path.read().split("\n")
          for st_fcm_file in ts_fcm_files:
            if not st_fcm_file: continue
            ts_parts = st_fcm_file.split(",")
            if ts_parts[0] == 'fcs_filename' and ts_parts[1] == 'label': continue
            self._ma_labels[ts_parts[0]] = ts_parts[1]
      else: 
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_label)


    def read_markers(self, path_to_marker):
      """
        Read markers and store into list  

        Params:
          path_to_marker: path to marker file
      """
      if os.path.exists(path_to_marker):
        with open(path_to_marker, "r") as oj_path:
          ts_markers = oj_path.read().split("\n")[0].split(",")
          self._ts_markers = [st_marker for st_marker in ts_markers if st_marker]
      else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_label)

  
    def read_flowdata(self, path_to_file, *args, **kwargs):
      """
        Aims:

        Params:
          path_to_file: path to file fcs 
          markers: list of selected markers  

        Returns: 
      """
      # st_fupath = os.path.join(path_to_dir, path_to_file) 
      print("Coding:", path_to_file)
      oj_f      = flowio.FlowData(path_to_file)
      ar_events = np.reshape(oj_f.events, (-1, oj_f.channel_count))
  
      ts_channels = []
      for i in range(1, oj_f.channel_count+1):
        key = str(i)
        if 'PnS' in oj_f.channels[key] and oj_f.channels[key]['PnS'] != u' ':
          ts_channels.append(oj_f.channels[key]['PnS'])
        elif 'PnN' in oj_f.channels[key] and oj_f.channels[key]['PnN'] != u' ':
          ts_channels.append(oj_f.channels[key]['PnN'])
        else:
          ts_channels.append('None')

      return ar_events, ts_channels
  
    ### def load_data(path_to_dir)
    ###   """
    ###     Aims: read the files from given directory
  
    ###     Params:
    ###       path_to_dir: path to directory where files are located
    ###   """
  
    ###   ts_files = os.listdir(path_to_dir)
    ###   if not ts_files:
    ###     print("[FATAL]: no files in %s" %path_to_dir)
    ###     sys.exit(0)
  
    ###   for st_file in ts_files:
    ###     if st_file.endswith(".fcs"):

def test():
  path_to_dir    = "./data/gated_NK/" 
  path_to_label  = "./data/NK_fcs_samples_with_labels.csv"
  path_to_marker = "./data/nk_marker.csv"
  o_fc_data = FCdata(path_to_dir, path_to_label, path_to_marker) 



test()


