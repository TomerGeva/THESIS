# **********************************************************************************************************************
# This file holds the values of the global variables, which are needed throughout the operation
# **********************************************************************************************************************
import numpy as np
from global_const import activation_type_e, pool_e, mode_e, model_output_e
from global_struct import ConvBlockData, AdaPadPoolData, PadPoolData, FCBlockData, ResFCBlockData, EdgeConvData, SetAbstractionData

# ==================================================================================================================
# Database Variables
# ==================================================================================================================
XRANGE = np.array([0, 4])  # np.array([0, 19])  # Range of the x coordinate of the structure in micro-meters
YRANGE = np.array([0, 4])  # np.array([0, 19])  # Range of the y coordinate of the structure in micro-meters

COORD_MEAN  = (XRANGE[1] + XRANGE[0]) / 2
COORD_SCALE = np.sqrt(2 * (XRANGE[1] - COORD_MEAN)**2)

NUM_OF_POINTS = 20

XQUANTIZE = 600  # 800  # 2500  # number of quantization points in the X coordinate
YQUANTIZE = 600  # 800  # 2500  # number of quantization points in the Y coordinate

DMIN = 0.1  # minimal allowed distance between cylinder centers, in micro-meters

SEED = 140993
# ==================================================================================================================
# DATALOADER HYPER-PARAMETERS
# ==================================================================================================================
NORM_SENS   = True
SENS_MEAN   = 0    # 1655  # 64458    # output normalization factor - mean sensitivity
SENS_STD    = 100  # 385   # 41025
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
PATH_DATABASE_TRAIN = ['..\\..\\databases\\corner_20_db_53k_20scat_signed_distance_lt_4e+01_train.csv',
                       '..\\..\\databases\\corner_20_db_53k_20scat_signed_distance_gt_4e+01_train.csv',
                       '..\\..\\databases\\corner_20_db_53k_20scat_signed_distance_gt_8e+01_train.csv',
                       '..\\..\\databases\\corner_20_db_53k_20scat_signed_distance_gt_1e+02_train.csv']
PATH_DATABASE_TEST  = ['..\\..\\databases\\corner_20_db_53k_20scat_signed_distance_lt_4e+01_test.csv',
                       '..\\..\\databases\\corner_20_db_53k_20scat_signed_distance_gt_4e+01_test.csv',
                       '..\\..\\databases\\corner_20_db_53k_20scat_signed_distance_gt_8e+01_test.csv',
                       '..\\..\\databases\\corner_20_db_53k_20scat_signed_distance_gt_1e+02_test.csv']
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
EMBED_DIM       = 2048
CONCAT_EDGECONV = True  # boolean stating if we want to concatenate all the edgeconv results at the end
FLATTEN_TYPE    = 'both'  # max, avg, both
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
EPOCH_NUM        = 101
LR               = 5e-5  # learning rate
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
    ['conv1d', ConvBlockData(1024, 1024, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['adapool1d', AdaPadPoolData(pool_e.AVG, pad=0, out_size=1)],
    ['linear', FCBlockData(512, in_neurons=1024, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear', FCBlockData(128, in_neurons=512, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear', FCBlockData(1,   in_neurons=128, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],
]
DGCNN_TOPOLOGY = [
    ['edgeconv', EdgeConvData(k=20, conv_data=ConvBlockData(in_channels=2*20,  out_channels=64,  kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
    ['edgeconv', EdgeConvData(k=20, conv_data=ConvBlockData(in_channels=64*2,  out_channels=128,  kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
    ['edgeconv', EdgeConvData(k=20, conv_data=ConvBlockData(in_channels=128*2, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
    ['edgeconv', EdgeConvData(k=20, conv_data=ConvBlockData(in_channels=256*2, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
    ['edgeconv', EdgeConvData(k=20, conv_data=ConvBlockData(in_channels=512*2, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
    ['conv1d', ConvBlockData(1472, EMBED_DIM, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2)],
    ['linear', FCBlockData(512, in_neurons=EMBED_DIM*2, bias=False, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU, alpha=0.2)],
    ['linear', FCBlockData(256, in_neurons=512, bias=False, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU, alpha=0.2)],
    ['res-linear', ResFCBlockData(1, in_neurons=256, layers=4, bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],
    # ['linear', FCBlockData(1,   in_neurons=256, bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)]
]
MODGCNN_TOPOLOGY = [
    ['modedgeconv', EdgeConvData(k=20, conv_data=ConvBlockData(in_channels=4, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
    ['sg_pointnet', SetAbstractionData(ntag=NUM_OF_POINTS/4, radius=0.25, k=None, in_channel=2+64, out_channels=[64, 64], pnet_kernel=1, residual=True)],
    ['modedgeconv', EdgeConvData(k=20, conv_data=ConvBlockData(in_channels=64*2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
    ['sg_pointnet', SetAbstractionData(ntag=NUM_OF_POINTS/8, radius=0.75, k=None, in_channel=2+64, out_channels=[64, 64], pnet_kernel=1, residual=True)],
    ['modedgeconv', EdgeConvData(k=10, conv_data=ConvBlockData(in_channels=64*2, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
    ['sg_pointnet', SetAbstractionData(ntag=NUM_OF_POINTS/16, radius=1.25, k=None, in_channel=2+128, out_channels=[128, 128], pnet_kernel=1, residual=True)],
    ['modedgeconv', EdgeConvData(k=10, conv_data=ConvBlockData(in_channels=128*2, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
    ['conv1d', ConvBlockData(512, EMBED_DIM, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2)],
    ['linear', FCBlockData(512, in_neurons=EMBED_DIM*2, bias=False, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU, alpha=0.2)],
    ['res-linear', ResFCBlockData(128, in_neurons=512, layers=3, bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.lReLU, alpha=0.2)],
    ['linear', FCBlockData(1, in_neurons=128, bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],
]
# DGCNN_TOPOLOGY = [
#     ['edgeconv', EdgeConvData(k='all', conv_data=ConvBlockData(in_channels=4,     out_channels=16, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
#     ['edgeconv', EdgeConvData(k='all', conv_data=ConvBlockData(in_channels=16*2,  out_channels=16, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
#     ['edgeconv', EdgeConvData(k='all', conv_data=ConvBlockData(in_channels=16*2,  out_channels=32, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
#     ['edgeconv', EdgeConvData(k='all', conv_data=ConvBlockData(in_channels=32*2,  out_channels=64, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2), aggregation='sum')],
#     ['conv1d', ConvBlockData(128, EMBED_DIM, kernel_size=1, stride=1, padding=0, bias=False, batch_norm=True, activation=activation_type_e.lReLU, alpha=0.2)],
#     ['linear', FCBlockData(512, in_neurons=EMBED_DIM*2, bias=False, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU, alpha=0.2)],
#     ['linear', FCBlockData(256, in_neurons=512, bias=False, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU, alpha=0.2)],
#     ['linear', FCBlockData(1,   in_neurons=256, bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)]
# ]
# DGCNN_TOPOLOGY = [
#     ['conv2d', ConvBlockData(2, 64, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
#     ['conv2d', ConvBlockData(64, 64, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
#     ['conv2d', ConvBlockData(64, 64, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
#     ['conv2d', ConvBlockData(64, 128, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
#     ['conv2d', ConvBlockData(128, 1024, 1, 1, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.ReLU)],
#     ['pool1d', PadPoolData(pool_e.MAX, pad=0, kernel=1024)],
# ]
