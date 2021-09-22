# ***************************************************************************************************
# This file holds the values of the global variables, which are needed throughout the operation
# ***************************************************************************************************
import numpy as np
from global_const import activation_type_e, pool_e

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
SIGNED_SENS = True
IMG_CHANNELS   = 1
MIXUP_FACTOR   = 0.3  # mixup parameter for the data
NUM_WORKERS    = 4

# ---logdir for saving the database ---
SAVE_PATH_DB = './database.pth'
# ---logdir for saving the network ----
SAVE_PATH_NET = './trained_nn.pth'
# -------------- paths --------------
PATH          = 'C:\\Users\\tomer\\Documents\\MATLAB\\csv_files\\grid_size_2500_2500\\corner_1450'
# \corner_1450_db_trunc.csv'  # \corner_1450_db_15p9k.csv'  # corner_1450_10k.csv' # corner_1450_db_17p9k
PATH_DATABASE_TRAIN = ['..\\..\\databases\\corner_1450_db_30p5k_unsigned_lt_1e+05_train.csv',
                       '..\\..\\databases\\corner_1450_db_30p5k_unsigned_gt_1e+05_train.csv',
                       '..\\..\\databases\\corner_1450_db_30p5k_unsigned_gt_3e+05_train.csv']
PATH_DATABASE_TEST  = ['..\\..\\databases\\corner_1450_db_30p5k_unsigned_lt_1e+05_test.csv',
                       '..\\..\\databases\\corner_1450_db_30p5k_unsigned_gt_1e+05_test.csv',
                       '..\\..\\databases\\corner_1450_db_30p5k_unsigned_gt_3e+05_test.csv']
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
BETA             = 2.44e-5       # the KL coefficient in the cost function
MSE_GROUP_WEIGHT = [1, 1.5, 12]  # weighted MSE according to sensitivity group
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

# ============================================================
# Chosen topology
# ============================================================
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
#   9. pool type
#  10. pool padding
#  11. pool size
"""
# Fully connected:
"""
#   1. Out channels
#   2. batch_norm
#   3. drop_rate
#   4. activation
#   5. alpha
"""
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
DENSE_ENCODER_TOPOLOGY = [
    ['conv',       1, 6, 25, 25, 0, True, 0, activation_type_e.ReLU, 0],
    ['dense',      100, 6, 3, 1, 1, True, 0, activation_type_e.ReLU, 0],
    ['transition', 0.5,    3, 1, 1, True, 0, activation_type_e.ReLU, 0, pool_e.AVG, 0, 2],
    ['dense',      100, 6, 3, 1, 1, True, 0, activation_type_e.ReLU, 0],
    ['transition', 0.5,    3, 1, 1, True, 0, activation_type_e.ReLU, 0, pool_e.AVG, 0, 2],
    ['dense',      100, 6, 3, 1, 1, True, 0, activation_type_e.ReLU, 0],
    ['transition', 0.5,    3, 1, 1, True, 0, activation_type_e.ReLU, 0, pool_e.AVG, (0, 1, 1, 0), 2],
    ['dense',      100, 6, 3, 1, 1, True, 0, activation_type_e.ReLU, 0],
    ['transition', 0.5,    3, 1, 1, True, 0, activation_type_e.ReLU, 0, pool_e.AVG, (0, 1, 1, 0), 2],
    ['dense',      100, 6, 3, 1, 1, True, 0, activation_type_e.ReLU, 0],
    ['transition', 0.5,    3, 1, 1, True, 0, activation_type_e.ReLU, 0, pool_e.AVG, (0, 1, 1, 0), 2],
    ['dense',      100, 6, 3, 1, 1, True, 0, activation_type_e.ReLU, 0],
    ['transition', 0.5,    3, 1, 0, True, 0, activation_type_e.ReLU, 0, pool_e.AVG, 0, 1],
    ['linear', 500,                  False, 0, activation_type_e.ReLU, 0],
    ['linear', 150,                  False, 0, activation_type_e.ReLU, 0],
    ['linear', 2 * LATENT_SPACE_DIM, False, 0, activation_type_e.null, 0]  # DO NOT CHANGE THIS LINE EVER
]

# --------------------------------------------------------
# Decoder topology
# --------------------------------------------------------
DECODER_TOPOLOGY = [
    ['linear',      300, False, 0, activation_type_e.ReLU, 0],
    ['linear',      100, False, 0, activation_type_e.ReLU, 0],
    ['linear',       25, False, 0, activation_type_e.ReLU, 0],
    ['linear_last',   1, False, 0, activation_type_e.null, 0],
]
