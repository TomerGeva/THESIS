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
PATH_DATABASE = 'C:\\Users\\tomer\\Documents\\MATLAB\\results\\grid_size_2500_2500\\corner_1450\\corner_1450_db.csv'
PATH_LOGS     = 'C:\\Users\\tomer\\Documents\\Thesis_logs'
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
CONV_DESCRIPTION = {0: 'conv',
                    1: 'conv',
                    2: 'conv',
                    3: 'pool',
                    4: 'conv',
                    5: 'conv',
                    6: 'pool',
                    7: 'conv',
                    8: 'pool',
                    9: 'conv'}
# Number of filters in each filter layer
FILTER_NUM   = [6,  # first layer
                16,  # second layer
                16,  # third layer
                16,  # fourth layer
                32,  # fifth layer
                32,  # sixth layer
                64,  # seventh layer
                ]
# Filter sizes for each filter layer
KERNEL_SIZE  = [25,  # first layer
                7,  # second layer
                7,  # third layer
                7,  # fourth layer
                7,  # fifth layer
                3,  # sixth layer
                3,  # seventh layer
                ]
# Stride values of the convolution layers
STRIDES      = [25,  # first layer
                1,  # second layer
                1,  # third layer
                1,  # fourth layer
                1,  # fifth layer
                1,  # sixth layer
                1,  # seventh layer
                ]
# Padding values of the convolution layers
PADDING      = [0,  # first layer
                0,  # second layer
                0,  # third layer
                0,  # fourth layer
                0,  # fifth layer
                0,  # sixth layer
                0,  # seventh layer
                ]
# Max pool size
MAX_POOL_SIZE   = 2
# FC layer sizes
FC_LAYERS = [1500,
             150,
             25,
             1,
             ]
