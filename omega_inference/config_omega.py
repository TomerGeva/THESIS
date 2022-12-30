# **********************************************************************************************************************
# This file holds the values of the global variables, which are needed throughout the operation
# **********************************************************************************************************************
import numpy as np
from global_struct import FCBlockData, ResFCBlockData
from global_const import activation_type_e, pool_e

# ==================================================================================================================
# Database Variables
# ==================================================================================================================
XRANGE = np.array([-10, 10])  # Range of the x coordinate of the structure in micro-meters
YRANGE = np.array([-10, 10])  # Range of the y coordinate of the structure in micro-meters

XQUANTIZE = 2500  # number of quantization points in the X coordinate
YQUANTIZE = 2500  # number of quantization points in the Y coordinate

NUM_OF_POINTS         = 20  # 500
NUM_OF_SAMPLED_POINTS = 20

OMEGA_FACTOR = 1e3  # The omegas are in reange [-1e-2, 1e-2], factoring to [-1,1]

DMIN = 0.1  # minimal allowed distance between cylinder centers, in micro-meters

SEED = 140993
# ==================================================================================================================
# DATALOADER HYPER-PARAMETERS
# when the grid is '0' for cylinder absence, and '1' for cylinder present,
# these are the std and mean for 1450 cylinders, need to normalize
# ==================================================================================================================
NUM_WORKERS   = 8
BATCH_SIZE    = 64
OPTIMIZE_TIME = True
# ==================================================================================================================
# Paths
# ==================================================================================================================

PATH_DATABASE_TRAIN = 'C:\\Users\\TomerG\\PycharmProjects\\THESIS_TG\\databases\\center_500_1_database_train.csv'
PATH_DATABASE_TEST  = 'C:\\Users\\TomerG\\PycharmProjects\\THESIS_TG\\databases\\center_500_1_database_test.csv'
PATH_LOGS           = 'C:\\Users\\TomerG\\PycharmProjects\\THESIS_TG\\results_omega'
# --------------------------------------------------------------------------------------------------------------
# Post processing paths
# --------------------------------------------------------------------------------------------------------------
FIG_DIR = 'figures'
PP_DATA = 'post_processing'
# ==================================================================================================================
# Global variables of the net - target RMS = 41000
# ==================================================================================================================
INIT_WEIGHT_MEAN = 0                     # weight init mean
INIT_WEIGHT_STD  = 0.02                  # weight init std
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Trainer configurations
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
EPOCH_NUM        = 301
LR               = 1e-5  # learning rate
SCHEDULER_STEP   = 40
SCHEDULER_GAMMA  = 0.5
MOM              = 0.9   # momentum update
GRAD_CLIP        = 5
SAVE_PER_EPOCH   = 40
# --------------------------------------------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------------------------------------------
FC_TOPOLOGY = [
    ['linear',      FCBlockData(256, in_neurons=2*NUM_OF_SAMPLED_POINTS, bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.lReLU)],
    # ['res-linear',  ResFCBlockData(512, layers=3, bias=True, batch_norm=True,  dropout_rate=0, activation=activation_type_e.lReLU)],
    # ['res-linear',  ResFCBlockData(128, layers=3, bias=True, batch_norm=True,  dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear', FCBlockData(128,  bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear', FCBlockData(64,  bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear', FCBlockData(1, bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],   # DO NOT CHANGE THIS LINE EVER
]
