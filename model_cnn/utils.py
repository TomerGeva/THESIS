from ConfigCNN import *


def get_topology():
    if MODEL_TYPE == 'vgg':
        return VGG_TOPOLOGY
    elif MODEL_TYPE == 'res-vgg':
        return VGG_RES_TOPOLOGY
    elif MODEL_TYPE == 'dense':
        return DENSE_TOPOLOGY
    elif MODEL_TYPE == 'seperable':
        return SEPARABLE_TOPOLOGY
    else:
        raise ValueError('MODEL TYPE is not set to a valid value')

