# ***************************************************************************************************
# This file holds the values of the global variables, which are needed throughout the operation
# ***************************************************************************************************
import numpy as np
from global_const import activation_type_e, pool_e
from global_struct import ConvBlockData, DenseBlockData, TransBlockData, FCBlockData

# ===================================
# Database Variables
# ===================================
XRANGE = np.array([0, 19])
YRANGE = np.array([0, 19])

XQUANTIZE = 2500
YQUANTIZE = 2500

# ========================================================================================
# when the grid is '0' for cylinder absence, and '1' for cylinder present,
# these are the std and mean for 1450 cylinders, need to normalize
# ========================================================================================
GRID_MEAN   = 0.000232
GRID_STD    = 0.015229786
SENS_MEAN   = 64458    # output normalization factor - mean sensitivity
SENS_STD    = 41025
ABS_SENS    = True
IMG_CHANNELS   = 1
MIXUP_FACTOR   = 0.3  # mixup parameter for the data
MIXUP_PROB     = 0.1  # mixup probability
NUM_WORKERS    = 8

# ---logdir for saving the database ---
SAVE_PATH_DB = './database.pth'
# ---logdir for saving the network ----
SAVE_PATH_NET = './trained_nn.pth'
# -------------- paths --------------
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
# ==================================
# Flow Control Variables
# ==================================
gather_DB = False
train     = True

# ============================================================
# Global variables of the net - target RMS = 41000
# ============================================================
# --------------------------------------------------------
# Hyper parameters
# --------------------------------------------------------
BETA             = 2.44e-5          # the KL coefficient in the cost function
MSE_GROUP_WEIGHT = [1, 1.5, 2, 12]  # weighted MSE according to sensitivity group
EPOCH_NUM        = 80
LR               = 1e-4  # learning rate
SCHEDULER_STEP   = 20
SCHEDULER_GAMMA  = 0.5
MOM              = 0.9   # momentum update
BATCH_SIZE       = 64

LATENT_SPACE_DIM = 50    # number of dimensions in the latent space
INIT_WEIGHT_MEAN = 0
INIT_WEIGHT_STD  = 0.02
GRAD_CLIP        = 5


# --------------------------------------------------------
# Encoder topology
# --------------------------------------------------------
ENCODER_TOPOLOGY = [
    ['conv', IMG_CHANNELS,   6, 25, 25, 0],  # conv layer: input channels, output channels, kernel, stride, padding
    ['conv',            6,  16,  5,  1, 2],
    ['pool', 2, 0],                             # pool layer: kernel
    ['conv',           16,  32,  5,  1, 2],
    ['pool', 2, 0],
    ['conv',           32,  64,  4,  1, 1],
    ['pool', 2, 0],
    ['conv',           64, 128,  3,  1, 0],
    ['conv',          128, 256,  3,  1, 0],
    ['pool', 2, 0],
    ['conv',          256, 512,  4,  1, 0],
    ['linear', 200],                         # linear layer: neuron number
    ['linear', 150],
    ['linear_last', 2 * LATENT_SPACE_DIM]
]
# --------------------------------------------------------
# Dense Encoder topology
# --------------------------------------------------------
# Init layer:
"""
#   1. in channels
#   2. out channels
#   3. kernel size
#   4. stride
#   5. padding
#   6. batch_norm
#   7. drop_rate
#   8. activation
#   9. alpha
"""
# Dense block:
"""
#   1. growth rate
#   2. depth
#   3. kernel size
#   4. stride
#   5. padding
#   6. batch_norm
#   7. drop_rate
#   8. activation
#   9. alpha
"""
# Transition:
"""
#   1. reduction rate
#   2. conv kernel
#   3. conv stride
#   4. conv padding
#   5. batch_norm
#   6. drop_rate
#   7. activation
#   8. alpha
#   9. pool padding
#  10. pool size
"""
# Fully connected:
"""
#   1. Out channels
#   2. batch_norm
#   3. drop_rate
#   4. activation
#   5. alpha
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
    ['linear', FCBlockData(150,                  batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['linear', FCBlockData(2 * LATENT_SPACE_DIM, batch_norm=False, dropout_rate=0, activation=activation_type_e.null)],  # DO NOT CHANGE THIS LINE EVER
]

# --------------------------------------------------------
# Decoder topology
# --------------------------------------------------------
"""
Decoder input: 2500 X 2500
conv1: 2500 --> 100
DECODER
"""
DECODER_TOPOLOGY = [
    ['linear', FCBlockData(300, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['linear', FCBlockData(100, batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['linear', FCBlockData(25,  batch_norm=True, dropout_rate=0, activation=activation_type_e.ReLU)],
    ['linear', FCBlockData(1,   batch_norm=False, dropout_rate=0, activation=activation_type_e.null)]  # DO NOT CHANGE THIS LINE EVER

]
