from ConfigDG import *
from database_functions import ModelManipulationFunctions
from LoggerDG import LoggerDG
from ScatCoord_DG import import_data_sets_coord
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PointNet import ModPointNet
import numpy as np
from auxiliary_functions import weighted_mse, _init_
import sklearn.metrics as metrics

def main():
    mmf = ModelManipulationFunctions()
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logdir = _init_(PATH_LOGS)
    logger = LoggerDG(logdir=logdir)
    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ================================================================================
    # Importing the database
    # ================================================================================
    train_loader, test_loaders, thresholds = import_data_sets_coord(PATH_DATABASE_TRAIN,
                                                                    PATH_DATABASE_TEST,
                                                                    BATCH_SIZE,
                                                                    abs_sens=ABS_SENS,
                                                                    num_workers=NUM_WORKERS
                                                                    )
    # ================================================================================
    # Creating the model
    # ================================================================================
    mod_pnet = ModPointNet(device, POINTNET_TOPOLOGY)
    mmf.initialize_weights(mod_pnet, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD)
    mod_pnet.to(device)
    # ================================================================================
    # Creating the trainer
    # ================================================================================







