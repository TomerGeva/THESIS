# **********************************************************************************************************************
# This file holds the values of the global variables, which are needed throughout the operation
# **********************************************************************************************************************
import numpy as np
from global_const import activation_type_e, pool_e, mode_e, model_output_e
from global_struct import ConvBlockData, DenseBlockData, TransBlockData, FCBlockData, ConvTransposeBlockData, PadPoolData

# ==================================================================================================================
# Database Variables
# ==================================================================================================================
XRANGE = np.array([0, 19])  # Range of the x coordinate of the structure in micro-meters
YRANGE = np.array([0, 19])  # Range of the y coordinate of the structure in micro-meters

XQUANTIZE = 2500  # number of quantization points in the X coordinate
YQUANTIZE = 2500  # number of quantization points in the Y coordinate

DMIN = 0.1  # minimal allowed distance between cylinder centers, in micro-meters

SEED = 140993
# ==================================================================================================================
# DATALOADER HYPER-PARAMETERS
# when the grid is '0' for cylinder absence, and '1' for cylinder present,
# these are the std and mean for 1450 cylinders, need to normalize
# ==================================================================================================================
DILATION    = 4
GRID_MEAN   = 0.000232
GRID_STD    = 0.015229786
SENS_MEAN   = 64458    # output normalization factor - mean sensitivity
SENS_STD    = 41025
ABS_SENS    = True
IMG_CHANNELS   = 1
MIXUP_FACTOR   = 0.3  # mixup parameter for the data
MIXUP_PROB     = 0  # mixup probability
NUM_WORKERS    = 8
BATCH_SIZE     = 64


# ==================================================================================================================
# Paths
# ==================================================================================================================
PATH          = 'C:\\Users\\tomer\\Documents\\MATLAB\\csv_files\\grid_size_2500_2500\\corner_1450'
# \corner_1450_db_trunc.csv'  # \corner_1450_db_15p9k.csv'  # corner_1450_10k.csv' # corner_1450_db_17p9k
PATH_DATABASE_TRAIN = ['..\\..\\databases\\corner_1450_db_30p5k_signed_lt_1e+05_train.csv',
                       '..\\..\\databases\\corner_1450_db_30p5k_signed_gt_1e+05_train.csv',
                       '..\\..\\databases\\corner_1450_db_30p5k_signed_gt_2e+05_train.csv',
                       '..\\..\\databases\\corner_1450_db_30p5k_signed_gt_3e+05_train.csv']
PATH_DATABASE_TEST  = ['..\\..\\databases\\corner_1450_db_30p5k_signed_lt_1e+05_test.csv',
                       '..\\..\\databases\\corner_1450_db_30p5k_signed_gt_1e+05_test.csv',
                       '..\\..\\databases\\corner_1450_db_30p5k_signed_gt_2e+05_test.csv',
                       '..\\..\\databases\\corner_1450_db_30p5k_signed_gt_3e+05_test.csv']
PATH_LOGS           = 'C:\\Users\\TomerG\\PycharmProjects\\THESIS_TG\\results'
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
# MODEL_OUT        = model_output_e.SENS
# MODE             = mode_e.AUTOENCODER
MODEL_OUT        = model_output_e.BOTH
MODE             = mode_e.VAE
LATENT_SPACE_DIM = 50                   # number of dimensions in the latent space
INIT_WEIGHT_MEAN = 0
INIT_WEIGHT_STD  = 0.02
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cost function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
BETA_DKL         = 1  # 2.44e-5          # the KL coefficient in the cost function
BETA_GRID        = 0.1
MSE_GROUP_WEIGHT = [1, 2, 2, 20]  # weighted MSE according to sensitivity group
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Trainer configurations
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
EPOCH_NUM        = 1001
LR               = 3e-4  # learning rate
SCHEDULER_STEP   = 50
SCHEDULER_GAMMA  = 0.9
MOM              = 0.9   # momentum update
GRAD_CLIP        = 5

# --------------------------------------------------------------------------------------------------------------
# Encoder topology
# --------------------------------------------------------------------------------------------------------------
DENSE_ENCODER_TOPOLOGY = [
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
VGG_ENCODER_TOPOLOGY = [
    ['conv', ConvBlockData(1, 16, 25, 25, 0, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],     # 2500  --> 100         LOS 25
    ['conv', ConvBlockData(16, 32, 3,  1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],     # 100   --> 100         LOS 25
    ['pool', PadPoolData(pool_e.AVG, pad=0, kernel=2)],                                                                 # 100   --> 50          LOS 50
    ['conv', ConvBlockData(32, 64, 3,  1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],     # 50    --> 50          LOS 50
    ['pool', PadPoolData(pool_e.AVG, pad=0, kernel=2)],                                                                 # 50    --> 25          LOS 100
    ['conv', ConvBlockData(64, 128, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],     # 25    --> 25          LOS 100
    ['pool', PadPoolData(pool_e.AVG, pad=(0, 1, 1, 0), kernel=2)],                                                      # 25    --> 26 --> 13   LOS 200
    ['conv', ConvBlockData(128, 256, 3,  1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 13    --> 13          LOS 200
    ['pool', PadPoolData(pool_e.AVG, pad=(0, 1, 1, 0), kernel=2)],                                                      # 13    --> 14 --> 7    LOS 400
    ['conv', ConvBlockData(256, 512, 3,  1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 7     --> 7           LOS 400
    ['pool', PadPoolData(pool_e.AVG, pad=0, kernel=7)],                                                                 # 7     --> 1           LOS 2500 + 300
    ['linear', FCBlockData(300,                  batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear', FCBlockData(2 * LATENT_SPACE_DIM, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],  # DO NOT CHANGE THIS LINE EVER
]
SEPARABLE_ENCODER_TOPOLOGY = [
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
"""
DENSE_ENCODER_TOPOLOGY = [
    ['conv',      ConvBlockData(1, 6, 25, 25, 0, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['dense',    DenseBlockData(100, 6, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition',  TransBlockData(0.5, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=0, pool_size=2)],
    ['dense',    DenseBlockData(100, 6, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition',  TransBlockData(0.5, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=0, pool_size=2)],
    ['dense',    DenseBlockData(100, 6, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition',  TransBlockData(0.5, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=(0, 1, 1, 0), pool_size=2)],
    ['dense',    DenseBlockData(100, 6, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition',  TransBlockData(0.5, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=(0, 1, 1, 0), pool_size=2)],
    ['dense',    DenseBlockData(100, 6, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition',  TransBlockData(0.5, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=(0, 1, 1, 0), pool_size=2)],
    ['dense',    DenseBlockData(100, 6, 3, 1, 1, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['transition',  TransBlockData(0.5, 3, 1, 0, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU, pool_type=pool_e.AVG, pool_pad=0, pool_size=1)],
    ['linear', FCBlockData(500,                  batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['linear', FCBlockData(180,                  batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['linear', FCBlockData(2 * LATENT_SPACE_DIM, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],  # DO NOT CHANGE THIS LINE EVER
]
"""

# --------------------------------------------------------------------------------------------------------------
# Decoder topology
# --------------------------------------------------------------------------------------------------------------
"""
DECODER_TOPOLOGY = [
    ['linear', FCBlockData(25, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear', FCBlockData(1, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],
]
"""
DECODER_TOPOLOGY = [
    ['linear',      FCBlockData(200, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear_last', FCBlockData(400, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['convTrans', ConvTransposeBlockData(400,  64, 11, 1, padding=0, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)], # 1   --> 11
    ['convTrans', ConvTransposeBlockData(64,   32,  6, 3, padding=2, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 11  --> 32
    ['convTrans', ConvTransposeBlockData(32,   16,  6, 3, padding=3, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 32  --> 93
    ['convTrans', ConvTransposeBlockData(16,    8,  6, 3, padding=2, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 93  --> 278
    ['convTrans', ConvTransposeBlockData(8,     4,  6, 3, padding=2, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 278 --> 833
    ['convTrans', ConvTransposeBlockData(4,     1,  6, 3, padding=1, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],   # 833 --> 2500 ; DO NOT CHANGE THIS LINE EVER
]
"""
DECODER_TOPOLOGY = [
    ['linear',      FCBlockData(200, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['linear_last', FCBlockData(400, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],
    ['convTrans', ConvTransposeBlockData(400, 64, 4, 1, padding=0, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 1   --> 4
    ['convTrans', ConvTransposeBlockData(64,  32, 6, 3, padding=2, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 4   --> 11
    ['convTrans', ConvTransposeBlockData(32,  32, 6, 3, padding=2, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 11  --> 32
    ['convTrans', ConvTransposeBlockData(32,  16, 6, 3, padding=3, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 32  --> 93
    ['convTrans', ConvTransposeBlockData(16,   8, 6, 3, padding=2, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 93  --> 278
    ['convTrans', ConvTransposeBlockData(8,    4, 6, 3, padding=2, batch_norm=True, dropout_rate=0, activation=activation_type_e.lReLU)],   # 278 --> 833
    ['convTrans', ConvTransposeBlockData(4,    1, 7, 3, padding=1, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],   # 833 --> 2500 ; DO NOT CHANGE THIS LINE EVER
]
"""
