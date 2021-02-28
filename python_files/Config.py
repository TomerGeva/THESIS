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

TRANSFORM_NORM = 0.5
IMG_CHANNELS   = 1

# ---path for saving the database ---
SAVE_PATH_DB = './database.pth'
# ---path for saving the network ----
SAVE_PATH_NET = './trained_nn.pth'
# -------------- paths --------------
PATH          = 'C:\\Users\\tomer\\Documents\\MATLAB\\csv_files\\grid_size_2500_2500\\corner_1450'
PATH_DATABASE = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\THESIS\\Git_repository\\databases\\corner_1450_10k.csv'
PATH_LOGS     = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\THESIS\\Thesis_logs'
# ==================================
# Flow Control Variables
# ==================================
gather_DB = False
train     = True

# ============================================================
# Global variables of the net
# ============================================================
# ----------------------
# Hyper parameters
# ----------------------
EPOCH_NUM        = 40
LR               = 3e-4  # learning rate
MU               = 0.9    # momentum update
NORM_FACT        = 3e4    # output normalization factor
BATCH_SIZE       = 32
INIT_WEIGHT_MEAN = 0
INIT_WEIGHT_STD  = 0.02
# ----------------------
# Network topology
# ----------------------
LAYER_DESCRIPTION = {0: 'conv',
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
FILTER_NUM       = [1,  # INPUT, do not change
                    6,  # first layer
                    16,  # second layer
                    16,  # third layer
                    16,  # fourth layer
                    16,  # fifth layer
                    16,  # sixth layer
                    32,  # seventh layer
                    ]
# Filter sizes for each filter layer
KERNEL_SIZE      = [25,  # first layer
                    5,  # second layer
                    5,  # third layer
                    4,  # fourth layer
                    3,  # fifth layer
                    3,  # sixth layer
                    4,  # seventh layer
                    ]
# Stride values of the convolution layers
STRIDES          = [25,  # first layer
                    1,  # second layer
                    1,  # third layer
                    1,  # fourth layer
                    1,  # fifth layer
                    1,  # sixth layer
                    1,  # seventh layer
                    ]
# Padding values of the convolution layers
PADDING          = [0,  # first layer
                    2,  # second layer
                    2,  # third layer
                    1,  # fourth layer
                    0,  # fifth layer
                    0,  # sixth layer
                    0,  # seventh layer
                    ]
# Max pool size
MAX_POOL_SIZE    = [2,  # first max-pool
                    2,  # second max-pool
                    2,  # third max-pool
                    2,  # fourth max-pool
                    ]
# FC layer sizes
FC_LAYERS = [150,
             25,
             1,
             ]
