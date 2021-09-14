# ***************************************************************************************************
# This file holds the values of the global variables, which are needed throughout the operation
# ***************************************************************************************************
import numpy as np

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
NUM_WORKERS    = 8

# ---logdir for saving the database ---
SAVE_PATH_DB = './database.pth'
# ---logdir for saving the network ----
SAVE_PATH_NET = './trained_nn.pth'
# -------------- paths --------------
PATH          = 'C:\\Users\\tomer\\Documents\\MATLAB\\csv_files\\grid_size_2500_2500\\corner_1450'
# \corner_1450_db_trunc.csv'  # \corner_1450_db_15p9k.csv'  # corner_1450_10k.csv' # corner_1450_db_17p9k
PATH_DATABASE_TRAIN = ['..\\..\\databases\\corner_1450_db_30k_unsigned_gt_1e+05_train.csv',
                       '..\\..\\databases\\corner_1450_db_30k_unsigned_gt_3e+05_train.csv',
                       '..\\..\\databases\\corner_1450_db_30k_unsigned_lt_1e+05_train.csv']
PATH_DATABASE_TEST  = ['..\\..\\databases\\corner_1450_db_30k_unsigned_gt_1e+05_test.csv',
                       '..\\..\\databases\\corner_1450_db_30k_unsigned_gt_3e+05_test.csv',
                       '..\\..\\databases\\corner_1450_db_30k_unsigned_lt_1e+05_test.csv']
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
BETA             = 1e-5     # the KL coefficient in the cost function
EPOCH_NUM        = 80
LR               = 2e-4     # learning rate
MOM              = 0.9      # momentum update
BATCH_SIZE       = 64
LATENT_SPACE_DIM = 50       # number of dimensions in the latent space
INIT_WEIGHT_MEAN = 0
INIT_WEIGHT_STD  = 0.02
GRAD_CLIP        = 5
SCHEDULER_STEP   = 20
SCHEDULER_GAMMA  = 0.5

# --------------------------------------------------------
# Encoder topology
# --------------------------------------------------------
"""
Encoder input: 2500 X 2500
conv1: 2500 --> 100
conv2: 100  --> 100
pool1: 100  --> 50
conv3: 50   --> 50
pool2: 50   --> 25
conv4: 25   --> 24
pool3: 24   --> 12
conv5: 10   --> 10
conv6: 10   --> 8
pool4: 8    --> 4
conv7: 4    --> 1

"""
"""
ENCODER_LAYER_DESCRIPTION = {0: 'conv',
                             1: 'conv',
                             2: 'pool',
                             3: 'conv',
                             4: 'pool',
                             5: 'conv',
                             6: 'pool',
                             7: 'conv',
                             8: 'conv',
                             9: 'pool',
                             10: 'conv',
                             11: 'linear',
                             12: 'linear',
                             13: 'linear last',
                             }
# Number of filters in each filter layer
ENCODER_FILTER_NUM       = [1,  # INPUT, do not change
                            6,   # first layer
                            16,  # second layer
                            16,  # third layer
                            16,  # fourth layer
                            16,  # fifth layer
                            32,  # sixth layer
                            64,  # seventh layer
                            ]
# Filter sizes for each filter layer
ENCODER_KERNEL_SIZE      = [25,  # first layer
                            5,  # second layer
                            5,  # third layer
                            4,  # fourth layer
                            3,  # fifth layer
                            3,  # sixth layer
                            4,  # seventh layer
                            ]
# Stride values of the convolution layers
ENCODER_STRIDES          = [25,  # first layer
                            1,  # second layer
                            1,  # third layer
                            1,  # fourth layer
                            1,  # fifth layer
                            1,  # sixth layer
                            1,  # seventh layer
                            ]
# Padding values of the convolution layers
ENCODER_PADDING          = [0,  # first layer
                            2,  # second layer
                            2,  # third layer
                            1,  # fourth layer
                            0,  # fifth layer
                            0,  # sixth layer
                            0,  # seventh layer
                            ]
# Max pool size
ENCODER_MAX_POOL_SIZE    = [2,  # first max-pool
                            2,  # second max-pool
                            2,  # third max-pool
                            2,  # fourth max-pool
                            ]
# FC layer sizes
ENCODER_FC_LAYERS = [150,
                     150,
                     2 * LATENT_SPACE_DIM,
                    ]
"""
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
    ['conv', 1, 6, 25, 25, 0],             # Init layer: in channels, out channels, kernel size, stride, padding
    ['dense', 100, 6, 3, 1, 1, 0],       # Dense block: growth rate, depth, kernel size, stride, padding, drop rate
    ['transition', 0.5, 3, 1, 1, 2, 0],    # Transition: reduction rate, conv kernel, conv stride, conv padding, pool size, pool padding
    ['dense', 100, 6, 3, 1, 1, 0],       # Fully connected: Out channels
    ['transition', 0.5, 3, 1, 1, 2, 0],
    ['dense', 100, 6, 3, 1, 1, 0],
    ['transition', 0.5, 3, 1, 1, 2, (0, 1, 1, 0)],
    ['dense', 100, 6, 3, 1, 1, 0],
    ['transition', 0.5, 3, 1, 1, 2, (0, 1, 1, 0)],
    ['dense', 100, 6, 3, 1, 1, 0],
    ['transition', 0.5, 3, 1, 1, 2, (0, 1, 1, 0)],
    ['dense', 100, 6, 3, 1, 1, 0],
    ['transition', 0.5, 3, 1, 0, 1, 0],
    ['linear', 500],
    ['linear', 150],
    ['linear_last', 2 * LATENT_SPACE_DIM]
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
    ['linear',      300],
    ['linear',      100],
    ['linear',       25],
    ['linear_last',   1],
]
