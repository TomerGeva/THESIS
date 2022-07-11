# **********************************************************************************************************************
# This file holds the values of the global variables, which are needed throughout the operation
# **********************************************************************************************************************
import numpy as np
from global_const import activation_type_e, pool_e, mode_e, model_output_e
from global_struct import ConvBlockData, AdaPadPoolData, PadPoolData, FCBlockData

# ==================================================================================================================
# Database Variables
# ==================================================================================================================
XRANGE = np.array([0, 4])  # np.array([0, 19])  # Range of the x coordinate of the structure in micro-meters
YRANGE = np.array([0, 4])  # np.array([0, 19])  # Range of the y coordinate of the structure in micro-meters

XQUANTIZE = 600  # 800  # 2500  # number of quantization points in the X coordinate
YQUANTIZE = 600  # 800  # 2500  # number of quantization points in the Y coordinate

DMIN = 0.1  # minimal allowed distance between cylinder centers, in micro-meters

SEED = 140993
# ==================================================================================================================
# DATALOADER HYPER-PARAMETERS
# ==================================================================================================================
NORM_SENS   = False
SENS_MEAN   = 1655  # 64458    # output normalization factor - mean sensitivity
SENS_STD    = 385   # 41025
# --------------------------------------------------------------------------------------------------------------
# Fully Connected based dataloader configurations
# --------------------------------------------------------------------------------------------------------------
COORD2MAP_SIGMA = 3  # deliminator in the coord2map function
N               = 1  # power of the gaussian
# --------------------------------------------------------------------------------------------------------------
# Common configurations
# --------------------------------------------------------------------------------------------------------------
ABS_SENS    = True
NUM_WORKERS    = 8
BATCH_SIZE     = 64
OPTIMIZE_TIME  = True


# ==================================================================================================================
# Paths
# ==================================================================================================================
PATH          = 'C:\\Users\\tomer\\Documents\\MATLAB\\csv_files\\grid_size_2500_2500\\corner_1450'
PATH_DATABASE_TRAIN = ['..\\..\\databases\\corner_500_db_30k_500scat_signed_lt_2e+03_train.csv',
                       '..\\..\\databases\\corner_500_db_30k_500scat_signed_gt_2e+03_train.csv',
                       '..\\..\\databases\\corner_500_db_30k_500scat_signed_gt_3e+03_train.csv',
                       '..\\..\\databases\\corner_500_db_30k_500scat_signed_gt_4e+03_train.csv']
PATH_DATABASE_TEST  = ['..\\..\\databases\\corner_500_db_30k_500scat_signed_lt_2e+03_test.csv',
                       '..\\..\\databases\\corner_500_db_30k_500scat_signed_gt_2e+03_test.csv',
                       '..\\..\\databases\\corner_500_db_30k_500scat_signed_gt_3e+03_test.csv',
                       '..\\..\\databases\\corner_500_db_30k_500scat_signed_gt_4e+03_test.csv']
PATH_LOGS           = 'C:\\Users\\TomerG\\PycharmProjects\\THESIS_TG\\results_dg'
# --------------------------------------------------------------------------------------------------------------
# Post processing paths
# --------------------------------------------------------------------------------------------------------------
FIG_DIR = 'figures'
PP_DATA = 'post_processing'

# ==================================================================================================================
# Global variables of the net - target RMS = 41000
# ==================================================================================================================
# --------------------------------------------------------------------------------------------------------------
# Hyper parameters
# --------------------------------------------------------------------------------------------------------------
EMBED_DIM = 1024
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Model configurations
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
INIT_WEIGHT_MEAN = 0                     # weight init mean when using the Gaussian init
INIT_WEIGHT_STD  = 0.02                  # weight init std
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cost function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MSE_GROUP_WEIGHT = [1, 2, 2, 4]  # [1, 2, 2, 20]  # weighted MSE according to sensitivity group
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Trainer configurations
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
EPOCH_NUM        = 1001
LR               = 2e-4  # learning rate
SCHEDULER_STEP   = 50
SCHEDULER_GAMMA  = 0.9
MOM              = 0.9   # momentum update
GRAD_CLIP        = 5

# --------------------------------------------------------------------------------------------------------------
# Model topology
# --------------------------------------------------------------------------------------------------------------
POINTNET_TOPOLOGY = [
    ['conv1d', ConvBlockData(2, 64, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['conv1d', ConvBlockData(64, 64, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['conv1d', ConvBlockData(64, 64, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['conv1d', ConvBlockData(64, 128, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['conv1d', ConvBlockData(128, 1024, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['adapool1d', AdaPadPoolData(pool_e.AVG, pad=0, out_size=1)],
    ['linear', FCBlockData(512, in_neurons=1024, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear', FCBlockData(1,   in_neurons=512, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],
]
# DGCNN_TOPOLOGY = [
#     ['conv2d', ConvBlockData(2, 64, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
#     ['conv2d', ConvBlockData(64, 64, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
#     ['conv2d', ConvBlockData(64, 64, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
#     ['conv2d', ConvBlockData(64, 128, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
#     ['conv2d', ConvBlockData(128, 1024, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
#     ['pool1d', PadPoolData(pool_e.MAX, pad=0, kernel=1024)],
# ]
