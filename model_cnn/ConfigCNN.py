# **********************************************************************************************************************
# This file holds the values of the global variables, which are needed throughout the operation
# **********************************************************************************************************************
import numpy as np
from global_const import activation_type_e, pool_e, mode_e, model_output_e, encoder_type_e
from global_struct import ConvBlockData, ResConvBlock2DData, DenseBlockData, TransBlockData, FCBlockData, ResFCBlockData,\
    ConvTransposeBlock2DData, PadPoolData, SelfAttentionData, EdgeConvData, SetAbstractionData

# ==================================================================================================================
# Database Variables
# ==================================================================================================================
XRANGE = np.array([0, 4])  # np.array([0, 19])  # Range of the x coordinate of the structure in micro-meters
YRANGE = np.array([0, 4])  # np.array([0, 19])  # Range of the y coordinate of the structure in micro-meters

XQUANTIZE = 600  # 800  # 2500  # number of quantization points in the X coordinate
YQUANTIZE = 600  # 800  # 2500  # number of quantization points in the Y coordinate

NUM_OF_POINTS = 500

DMIN = 0.1  # minimal allowed distance between cylinder centers, in micro-meters

SEED = 140993
# ==================================================================================================================
# DATALOADER HYPER-PARAMETERS
# when the grid is '0' for cylinder absence, and '1' for cylinder present,
# these are the std and mean for 1450 cylinders, need to normalize
# ==================================================================================================================
# --------------------------------------------------------------------------------------------------------------
# Convolution based dataloader configurations
# --------------------------------------------------------------------------------------------------------------
DILATION     = 4
NORM_GRID    = False
# Grid norm for 2D grids
GRID_MEAN    = 0.000232
GRID_STD     = 0.015229786
# Coord norm for point clouds
COORD_MEAN   = (XRANGE[1] + XRANGE[0]) / 2
COORD_SCALE  = np.sqrt(2 * (XRANGE[1] - COORD_MEAN)**2)
ABS_SENS     = True
NORM_SENS    = True
SENS_MEAN    = 0    # 1655  # 64458    # output normalization factor - mean sensitivity
SENS_STD     = 1e2  # 385   # 41025
IMG_CHANNELS = 1
NUM_WORKERS   = 8
BATCH_SIZE    = 64
OPTIMIZE_TIME = True
# --------------------------------------------------------------------------------------------------------------
# Fully Connected based dataloader configurations
# --------------------------------------------------------------------------------------------------------------
COORD2MAP_SIGMA = 3  # deliminator in the coord2map function
N               = 1  # power of the gaussian

# ==================================================================================================================
# Paths
# ==================================================================================================================
PATH          = 'C:\\Users\\tomer\\Documents\\MATLAB\\csv_files\\grid_size_2500_2500\\corner_1450'
PATH_DATABASE_TRAIN = ['..\\..\\databases\\corner_500_db_50k_500scat_signed_lt_2e+03_train.csv',
                       '..\\..\\databases\\corner_500_db_50k_500scat_signed_gt_2e+03_train.csv',
                       '..\\..\\databases\\corner_500_db_50k_500scat_signed_gt_3e+03_train.csv',
                       '..\\..\\databases\\corner_500_db_50k_500scat_signed_gt_4e+03_train.csv']
PATH_DATABASE_TEST  = ['..\\..\\databases\\corner_500_db_50k_500scat_signed_lt_2e+03_test.csv',
                       '..\\..\\databases\\corner_500_db_50k_500scat_signed_gt_2e+03_test.csv',
                       '..\\..\\databases\\corner_500_db_50k_500scat_signed_gt_3e+03_test.csv',
                       '..\\..\\databases\\corner_500_db_50k_500scat_signed_gt_4e+03_test.csv']
PATH_LOGS           = 'C:\\Users\\TomerG\\PycharmProjects\\THESIS_TG\\results_cnn'
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Model configurations
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODEL_TYPE       = "res-vgg"
INIT_WEIGHT_MEAN = 0                     # weight init mean
INIT_WEIGHT_STD  = 0.02                  # weight init std
EMBED_DIM        = 1024
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cost function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MSE_GROUP_WEIGHT = [1, 4, 4, 16]  # [1, 2, 2, 20]  # weighted MSE according to sensitivity group
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Trainer configurations
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
EPOCH_NUM        = 1501
LR               = 3e-4  # learning rate
SCHEDULER_STEP   = 50
SCHEDULER_GAMMA  = 0.9
MOM              = 0.9   # momentum update
GRAD_CLIP        = 5

# --------------------------------------------------------------------------------------------------------------
# Encoder topology
# --------------------------------------------------------------------------------------------------------------
DENSE_TOPOLOGY     = [
    ['conv',       ConvBlockData(1, 12, 25, 25, 0, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['dense',      DenseBlockData(64, 6, 3, 1, 1,  batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition', TransBlockData(0.5, 3, 1, 1,    batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=0, pool_size=2)],
    ['dense',      DenseBlockData(64, 6, 3, 1, 1,  batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition', TransBlockData(0.5, 3, 1, 1,    batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=0, pool_size=2)],
    ['dense',      DenseBlockData(64, 6, 3, 1, 1,  batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition', TransBlockData(0.5, 3, 1, 1,    batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=(0, 1, 1, 0), pool_size=2)],
    ['dense',      DenseBlockData(64, 6, 3, 1, 1,  batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition', TransBlockData(0.5, 3, 1, 1,    batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=(0, 1, 1, 0), pool_size=2)],
    ['dense',      DenseBlockData(64, 6, 3, 1, 1,  batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition', TransBlockData(0.5, 3, 1, 1,    batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=(0, 1, 1, 0), pool_size=7)],
    ['linear',     FCBlockData(300,                batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['linear',     FCBlockData(2 * LATENT_SPACE_DIM, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],  # DO NOT CHANGE THIS LINE EVER
]
SEPARABLE_TOPOLOGY = [
    ['conv',     ConvBlockData(1, 16, 25, 25, 0, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.lReLU)],     # 2500  --> 100         LOS 25
    ['sep-conv', ConvBlockData(16, 32, 3,  1, 1, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.lReLU)],     # 100   --> 100         LOS 25
    ['pool',     PadPoolData(pool_e.AVG, pad=0, kernel=2)],                                                                             # 100   --> 50          LOS 50
    ['sep-conv', ConvBlockData(32, 64, 3,  1, 1, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.lReLU)],     # 50    --> 50          LOS 50
    ['pool',     PadPoolData(pool_e.AVG, pad=0, kernel=2)],                                                                             # 50    --> 25          LOS 100
    ['sep-conv', ConvBlockData(64, 128, 3, 1, 1, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.lReLU)],     # 25    --> 25          LOS 100
    ['pool',     PadPoolData(pool_e.AVG, pad=(0, 1, 1, 0), kernel=2)],                                                                  # 25    --> 26 --> 13   LOS 200
    ['sep-conv', ConvBlockData(128, 256, 3,  1, 1, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.lReLU)],   # 13    --> 13          LOS 200
    ['pool',     PadPoolData(pool_e.AVG, pad=(0, 1, 1, 0), kernel=2)],                                                                  # 13    --> 14 --> 7    LOS 400
    ['sep-conv', ConvBlockData(256, 512, 3,  1, 1, batch_norm=True, bias=False, dropout_rate=0, activation=activation_type_e.lReLU)],   # 7     --> 7           LOS 400
    ['pool',     PadPoolData(pool_e.AVG, pad=0, kernel=7)],                                                                             # 7     --> 1           LOS 2500 + 300
    ['linear', FCBlockData(300,                  batch_norm=False, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear', FCBlockData(2 * LATENT_SPACE_DIM, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],  # DO NOT CHANGE THIS LINE EVER
]
# TRANS_ENCODER_TOPOLOGY = [
#         ['transformer', SelfAttentionData(patch_size_x=50, patch_size_y=50, embed_size=1250)],
#         ['linear', FCBlockData(1000, bias=False, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
#         ['linear', FCBlockData(500,  bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.lReLU)],
#         ['linear', FCBlockData(500,  bias=False, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
#         ['linear', FCBlockData(300,  bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.lReLU)],
#         ['linear', FCBlockData(300,  bias=False, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
#         ['linear', FCBlockData(2 * LATENT_SPACE_DIM, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],  # DO NOT CHANGE THIS LINE EVER
#     ]
TRANS_TOPOLOGY     = [
        ['transformer', SelfAttentionData(patch_size_x=50, patch_size_y=50, embed_size=1250)],
        ['linear',      FCBlockData(1000, bias=False, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
        ['res-linear',  ResFCBlockData(500, layers=3, bias=True, batch_norm=True,  dropout_rate=0, activation=activation_type_e.lReLU)],
        ['res-linear',  ResFCBlockData(300, layers=3, bias=True, batch_norm=True,  dropout_rate=0, activation=activation_type_e.lReLU)],
        ['linear', FCBlockData(2 * LATENT_SPACE_DIM,  bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],  # DO NOT CHANGE THIS LINE EVER
    ]
FC_TOPOLOGY        = [
    ['linear',      FCBlockData(1500, in_neurons=1000, batch_norm=False, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['res-linear',  ResFCBlockData(600, layers=4, bias=True, batch_norm=True,  dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear',      FCBlockData(2 * LATENT_SPACE_DIM,  bias=True, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],  # DO NOT CHANGE THIS LINE EVER
]
VGG_TOPOLOGY = [
    ['conv', ConvBlockData(1, 64, 12, 12, 0, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],       # 600   --> 50          LOS 12
    # ['conv', ConvBlockData(32, 32, 3,  1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],     # 100   --> 100         LOS 25
    # ['pool', PadPoolData(pool_e.AVG, pad=0, kernel=2)],                                                                 # 100   --> 50          LOS 50
    ['conv', ConvBlockData(64, 64, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],        # 50    --> 50          LOS 16
    ['pool', PadPoolData(pool_e.AVG, pad=0, kernel=2)],                                                                   # 50    --> 25          LOS 32
    ['conv', ConvBlockData(64, 256, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],       # 25    --> 25          LOS 32
    ['pool', PadPoolData(pool_e.AVG, pad=(0, 1, 1, 0), kernel=2)],                                                        # 25    --> 26 --> 13   LOS 64
    ['conv', ConvBlockData(256, 512, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],      # 13    --> 13          LOS 64
    ['pool', PadPoolData(pool_e.AVG, pad=(0, 1, 1, 0), kernel=2)],                                                        # 13    --> 14 --> 7    LOS 128
    ['conv', ConvBlockData(512, 1024, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],     # 7     --> 7           LOS 128
    ['pool', PadPoolData(pool_e.AVG, pad=0, kernel=7)],                                                                   # 7     --> 1           LOS 800 + 300
    ['linear', FCBlockData(300, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear', FCBlockData(2 * LATENT_SPACE_DIM, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],
    # DO NOT CHANGE THIS LINE EVER
]
VGG_RES_TOPOLOGY = [
    ['conv', ConvBlockData(1, 64, 12, 12, 0, batch_norm=True, dropout_rate=0.1, activation=activation_type_e.lReLU)],              # 600   --> 50          LOS 12
    # ['conv', ConvBlock2DData(32, 32, 3,  1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],            # 100   --> 100         LOS 25
    # ['pool', PadPool2DData(pool_e.AVG, pad=0, kernel=2)],                                                                        # 100   --> 50          LOS 50
    ['res-conv', ResConvBlock2DData(64, 64, 3, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],     # 50    --> 50          LOS 16
    ['pool',     PadPoolData(pool_e.AVG, pad=0, kernel=2)],                                                                      # 50    --> 25          LOS 32
    ['res-conv', ResConvBlock2DData(64, 256, 3, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],    # 25    --> 25          LOS 32
    ['pool',     PadPoolData(pool_e.AVG, pad=(0, 1, 1, 0), kernel=2)],                                                           # 25    --> 26 --> 13   LOS 64
    ['res-conv', ResConvBlock2DData(256, 512, 3, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 13    --> 13          LOS 64
    ['pool',     PadPoolData(pool_e.AVG, pad=(0, 1, 1, 0), kernel=2)],                                                           # 13    --> 14 --> 7    LOS 128
    ['res-conv', ResConvBlock2DData(512, 1024, 3, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],  # 7     --> 7           LOS 128
    ['pool',     PadPoolData(pool_e.AVG, pad=0, kernel=7)],                                                                      # 7     --> 1           LOS 800 + 300
    ['linear',   FCBlockData(1000, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear',   FCBlockData(2 * LATENT_SPACE_DIM, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],   # DO NOT CHANGE THIS LINE EVER
]

