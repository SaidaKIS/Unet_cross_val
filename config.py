import numpy as np
import os
import torch
from easydict import EasyDict as edict

cfg = edict()
cfg.file = '../GT.hdf5' # Complete database 
cfg.l = 100 #submap dataset
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.seq_len = 2 # Length of the temporal sequence from a center frame 
cfg.n_hidden = 16 # initial hidden layers
cfg.h = 96 #height of the box
cfg.w = 96 #width of the box
cfg.batch = 16
cfg.bin_classes = ['Intergranular lane', 'Granule', 'Exploding granule']
cfg.channels = 2 # initial channels
cfg.N_EPOCHS = 10
cfg.loss = 'mIoU' # 'CrossEntropy', 'FocalLoss', 'mIoU'
cfg.lr = 3e-4 #Learing rate - inicial 1e-3 test 3e-4
cfg.dropout = True



