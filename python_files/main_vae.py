# import numpy as np
# import os
import torch
from ConfigVAE                  import *
from LoggerVAE                  import LoggerVAE
from TrainerVAE                 import TrainerVAE
from ModVAE                     import ModVAE
from neural_network_functions   import initialize_weights
from ScatterCoordinateDataset   import import_data_sets


def main_vae():
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logger = LoggerVAE(logdir=PATH_LOGS)

    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ================================================================================
    # Importing the data
    # ================================================================================
    train_loader, test_loader = import_data_sets(BATCH_SIZE, 0.15)

    # ================================================================================
    # Creating the net & trainer objects
    # ================================================================================
    mod_vae = ModVAE(device=device)
    initialize_weights(mod_vae, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD)
    mod_vae.to(device)  # allocating the computation to the CPU or GPU

    trainer = TrainerVAE(mod_vae, lr=LR, mom=MOM, beta=BETA)

    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(mod_vae, train_loader, test_loader, logger, save_per_epochs=40)


if __name__ == '__main__':
    main_vae()
