# import numpy as np
# import os
import torch
from ConfigVAE                  import *
from LoggerVAE                  import LoggerVAE
from LoggerLatent               import LoggerLatent
from TrainerVAE                 import TrainerVAE
from TrainerLatent              import TrainerLatent
from ModVAE                     import ModVAE
from auxiliary_functions        import initialize_weights
from ScatterCoordinateDataset   import import_data_sets
from global_const               import encoder_type_e
from database_functions         import load_decoder


def main_vae(encoder_type=encoder_type_e.DENSE):
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
    train_loader, test_loaders, thresholds = import_data_sets(BATCH_SIZE,
                                                              mixup_factor=MIXUP_FACTOR,
                                                              mixup_prob=MIXUP_PROB,
                                                              abs_sens=ABS_SENS)

    # ================================================================================
    # Creating the net & trainer objects
    # ================================================================================
    if encoder_type == encoder_type_e.DENSE:
        mod_vae = ModVAE(device=device,
                         encoder_topology=DENSE_ENCODER_TOPOLOGY,
                         decoder_topology=DECODER_TOPOLOGY,
                         latent_space_dim=LATENT_SPACE_DIM,
                         encoder_type=encoder_type,
                         mode=MODE,
                         model_out=MODEL_OUT)
    elif encoder_type == encoder_type_e.VGG:
        mod_vae = ModVAE(device=device,
                         encoder_topology=ENCODER_TOPOLOGY,
                         decoder_topology=DECODER_TOPOLOGY,
                         latent_space_dim=LATENT_SPACE_DIM,
                         encoder_type=encoder_type)
    initialize_weights(mod_vae, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD)
    mod_vae.to(device)  # allocating the computation to the CPU or GPU

    trainer = TrainerVAE(mod_vae,
                         lr=LR,
                         mom=MOM,
                         beta_dkl=BETA_DKL,
                         beta_grid=BETA_GRID,
                         sched_step=SCHEDULER_STEP,
                         sched_gamma=SCHEDULER_GAMMA,
                         grad_clip=GRAD_CLIP,
                         group_thresholds=thresholds,
                         group_weights=MSE_GROUP_WEIGHT,
                         abs_sens=ABS_SENS)

    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(mod_vae, train_loader, test_loaders, logger, save_per_epochs=20)


def main_optim_latent(path=None):
    # ================================================================================
    # Loading the decoder creating the input vector TODO: need to set a path to a requested model folder
    # ================================================================================
    decoder, latent_dim = load_decoder()
    input_vec = torch.rand(latent_dim)
    # ================================================================================
    # Setting the logger - TODO: need to set a path to a requested model folder
    # ================================================================================
    logger = LoggerLatent(path=PATH_LOGS)
    # ================================================================================
    # Creating the trainer object
    # ================================================================================
    trainer = TrainerLatent(input_vec=input_vec,
                            lr=LR,
                            mom=MOM,
                            beta_dkl=BETA_DKL,
                            sched_step=SCHEDULER_STEP,
                            sched_gamma=SCHEDULER_GAMMA,
                            grad_clip=GRAD_CLIP,
                            abs_sens=ABS_SENS,
                            sens_std=SENS_STD,
                            sens_mean=SENS_MEAN)
    # ================================================================================
    # Training
    # ================================================================================
    trainer.optimize_input(input_vec, decoder, 2000, logger, save_per_epoch=1)


if __name__ == '__main__':
    phase = 1
    # ================================================================================
    # Training VAE on scatterer arrays and matching sensitivities
    # ================================================================================
    if phase == 1:
        enc_type = encoder_type_e.DENSE
        main_vae(enc_type)
    # ================================================================================
    # Using the decoder to maximize sensitivity prediction
    # ================================================================================
    if phase == 2:
        main_optim_latent()
