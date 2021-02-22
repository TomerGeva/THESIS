import os
# ==========================================================
# Constants
# ==========================================================
CONSTELLATION       = [-3, -1, 1, 3]
PRECURSOR_NUM       = 5
POSTCURSOR_NUM      = 20
X_SIZE              = PRECURSOR_NUM + POSTCURSOR_NUM + 1

# ===========================================================
# FLow Control
# ===========================================================
TRAIN_NEW_NET              = True   # change to perform training for a new net
LOAD_AND_TEST              = False  # change if you want to load a pre-trained model, and perform accuracy test

# ===========================================================
# dictionaries
# ===========================================================
LABELS2SYMBOLS_DICT = {0: -3,
                       1: -1,
                       2: 1,
                       3: 3}  # label dictionary
SYMBOLS2LABELS_DICT = {-3: 0,
                       -1: 1,
                       1: 2,
                       3: 3}  # symbol dictionary

# ==========================================================
# important paths
# ==========================================================
PATH_DATA   = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\Deep_Learning\\Final_Project\\data\\databases'
PATH_LOGS   = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\Deep_Learning\\Final_Project\\data\\logs'
PATH_MODELS = 'C:\\Users\\tgeva\\Documents\\UNIVERSITY\\Deep_Learning\\Final_Project\\data\\models'

PATH_TRAIN_DATA = os.path.join(PATH_DATA, "database_deep_learning_proj_prbs17.csv")
PATH_VAL_DATA   = os.path.join(PATH_DATA, "database_deep_learning_proj_prbs13.csv")

# ============================================================
# Global variables of the net
# ============================================================
# ----------------------
# Hyper parameters
# ----------------------
EPOCH_NUM        = 1000
MU               = 1e-3  # learning rate
MOMENTUM         = 0.9
BATCH_SIZE       = 128

# ----------------------
# Network topology
# ----------------------
LAYER_DESCRIPTION = {0: 'conv',
                     1: 'ReLU',
                     2: 'pool',
                     3: 'conv',
                     4: 'ReLU',
                     5: 'pool',
                     6: 'linear'}

# ++++++++++++++++++++++
# conv layers topology
# ++++++++++++++++++++++
# Number of filters in each filter layer
FILTER_NUM   = [1,   # input, DO NOT CHANGE
                32,  # first conv
                64,  # second conv
                ]
# Filter sizes for each filter layer
KERNEL_SIZE  = [5,  # first layer
                5,  # second layer
                ]
# Stride values of the convolution layers
STRIDES      = [1,  # first layer
                1,  # second layer
                ]
# Padding values of the convolution layers
PADDING      = [2,  # first conv
                2,  # second conv
                ]
# Max pool size
MAX_POOL_SIZE = [2,  # first conv
                 2,  # second conv
                 ]
# ++++++++++++++++++++++
# FC layer topology
# ++++++++++++++++++++++
FC_LAYERS = [40,  # first FC
             4,    # output FC
             ]
