'''
Written: Rojan Shrestha PhD
Mon Nov 18 17:35:38 2019
'''

import sys

import numpy as np

import flowio # pip install FlowIO

class FCdata:
  
  def __init__(self, path_to_dir, path_to_label_data, path_to_marker):
    """
      Aims: class to store
      Params:
        st_path_to_file: 
        st_path_to_label: 
        st_path_to_marker: 
    """
    self._ma_labels  = {}
    self._ma_data    = {}
    self._ts_markers = []

    self._Xtrains   = []
    self._Xvalids   = []
    self._Xtest     = []
    self._Ytrains   = []
    self._Yvalids   = []
    self._Ytest     = []


    self.read_labels(path_to_label)
    self.read_markers(path_to_marker)

    # get ...
    oj_idata = self.iData(path_to_dir)
    for st_path, st_label in self._ma_labels.items():
      ar_events, ts_channels = oj_idata.read_flowdata(st_path, 
                                                      markers = self._ts_markers, 
                                                      transform=None, 
                                                      auto_comp=False)
      self._ma_data[st_path] = ar_events


  def read_labels(self, path_to_label):
    """
      Aims: read the label of for each mass cytometry file 

      Params:
        path_to_label: path to label file
    """
    with open(path_to_label, "r") as oj_path:
      ts_fcm_files = oj_path.read().split("\n")
      for st_fcm_file in ts_fcm_files:
        if not st_fcm_file: continue
        ts_parts = st_fcm_file.split("\t")
        self._labels[ts_parts[0]] = ts_parts[1]

  def read_markers(self, path_to_marker)
    """
    """
    with open(path_to_marker, "r") as oj_path:
      ts_markers = oj_path.read().split("\n")
      self._ts_markers = [st_marker for st_marker in ts_markers if st_marker]

  def load_data(self): 
    """
      
    """
    in_num_sample   = len(self._ma_labels)
    in_train_sample = int(0.7*in_num_sample)
     

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


  class iData:
   
    def __init__(self, path_to_dir):
      self._path_to_dir = path_to_dir 

  
    def read_flowdata(self, path_to_file, markers = [], *args, **kwargs):
      """
        Aims:

        Params:
          path_to_file:
          markers: 
      """
      st_fupath = os.path.join(path_to_dir, path_to_file) 
      oj_f      = flowio.FlowData(st_fupath, args, kwargs)
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

      # if marker genes are given 
      if markers:
        ts_marker_idx = [ts_channels.index(st_marker) for st_marker in markers]
        return ar_events[:, ts_marker_idx], ts_channels 
      else:
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
 
  FCdata:



