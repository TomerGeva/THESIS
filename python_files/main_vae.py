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
    train_loader, test_loaders = import_data_sets(BATCH_SIZE, 0.15)

    # ================================================================================
    # Creating the net & trainer objects
    # ================================================================================
    mod_vae = ModVAE(device=device, encoder_topology=DENSE_ENCODER_TOPOLOGY, decoder_topology=DECODER_TOPOLOGY, latent_space_dim=LATENT_SPACE_DIM)
    initialize_weights(mod_vae, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD)
    mod_vae.to(device)  # allocating the computation to the CPU or GPU

    trainer = TrainerVAE(mod_vae, lr=LR, mom=MOM, beta=BETA, sched_step=SCHEDULER_STEP, sched_gamma=SCHEDULER_GAMMA, grad_clip=GRAD_CLIP)

    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(mod_vae, train_loader, test_loaders, logger, save_per_epochs=20)


if __name__ == '__main__':
    main_vae()
