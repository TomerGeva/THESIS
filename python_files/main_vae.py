# import numpy as np
# import os
import torch
from ConfigVAE                  import *
from LoggerVAE                  import LoggerVAE
from TrainerVAE                 import TrainerVAE
from ModVAE                     import ModVAE
from auxiliary_functions        import initialize_weights
from ScatterCoordinateDataset   import import_data_sets
from global_const               import encoder_type_e


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
                                                              abs_sens=SIGNED_SENS)

    # ================================================================================
    # Creating the net & trainer objects
    # ================================================================================
    if encoder_type == encoder_type_e.DENSE:
        mod_vae = ModVAE(device=device,
                         encoder_topology=DENSE_ENCODER_TOPOLOGY,
                         decoder_topology=DECODER_TOPOLOGY,
                         latent_space_dim=LATENT_SPACE_DIM,
                         encoder_type=encoder_type)
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
                         beta=BETA,
                         sched_step=SCHEDULER_STEP,
                         sched_gamma=SCHEDULER_GAMMA,
                         grad_clip=GRAD_CLIP,
                         group_thresholds=thresholds,
                         group_weights=MSE_GROUP_WEIGHT,
                         abs_sens=SIGNED_SENS)

    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(mod_vae, train_loader, test_loaders, logger, save_per_epochs=20)


if __name__ == '__main__':
    enc_type = encoder_type_e.DENSE
    main_vae(enc_type)
