# import numpy as np
# import os
import torch
import os
import matplotlib.pyplot as plt
from ConfigVAE                  import *
from LoggerVAE                  import LoggerVAE
from LoggerLatent               import LoggerLatent
from TrainerVAE                 import TrainerVAE
from TrainerLatent              import TrainerLatent
from ModVAE                     import ModVAE
from auxiliary_functions        import initialize_weights, plot_grid, get_full_path
from ScatterCoordinateDataset   import import_data_sets
from global_const               import encoder_type_e
from database_functions         import load_decoder
from database_functions         import load_state_train


def main_vae(encoder_type=encoder_type_e.DENSE, load_model=None, start_epoch=0):
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
                                                              abs_sens=ABS_SENS,
                                                              dilation=DILATION)
    if load_model is None:
        # ============================================================================
        # Creating the net & trainer objects
        # ============================================================================
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
                             encoder_topology=VGG_ENCODER_TOPOLOGY,
                             decoder_topology=DECODER_TOPOLOGY,
                             latent_space_dim=LATENT_SPACE_DIM,
                             encoder_type=encoder_type,
                             mode=MODE,
                             model_out=MODEL_OUT                         )
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
    else:
        # ==============================================================================
        # Extracting the full file path
        # ==============================================================================
        chosen_file = get_full_path(load_model, start_epoch)
        # ==============================================================================
        # Loading the needed models and data
        # ==============================================================================
        mod_vae, trainer = load_state_train(chosen_file, thresholds=thresholds)

    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(mod_vae, train_loader, test_loaders, logger, save_per_epochs=20)


def main_optim_input(path=None, epoch=None):
    # ================================================================================
    # creating full file path
    # ================================================================================
    chosen_file = get_full_path(path, epoch=epoch)
    # ================================================================================
    # Loading the decoder creating the input vector
    # ================================================================================
    mod_vae, _ = load_state_train(chosen_file)
    mod_vae.mode = mode_e.AUTOENCODER
    mod_vae.model_out = model_output_e.SENS
    mod_vae.decoder.model_out = model_output_e.SENS
    mod_vae.requires_grad_(False)

    input_mat  = torch.nn.Parameter(torch.randn([1, 1, XQUANTIZE, YQUANTIZE], device=mod_vae.device), requires_grad=True)
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logger = LoggerLatent(logdir=path, filename='logger_latent.txt')
    # ================================================================================
    # Creating the trainer object
    # ================================================================================
    trainer = TrainerLatent(input_vec=input_mat,
                            lr=LR,
                            mom=MOM,
                            beta=BETA_DKL,
                            sched_step=SCHEDULER_STEP,
                            sched_gamma=SCHEDULER_GAMMA,
                            grad_clip=GRAD_CLIP,
                            abs_sens=ABS_SENS,
                            sens_std=SENS_STD,
                            sens_mean=SENS_MEAN)
    # ================================================================================
    # Training
    # ================================================================================
    optim_mat = trainer.optimize_input(input_mat, mod_vae, 2000, logger, save_per_epoch=1)
    plot_grid(optim_mat)


def main_optim_latent(path=None, epoch=None):
    # ================================================================================
    # creating full file path
    # ================================================================================
    chosen_file = get_full_path(path, epoch=epoch)
    # ================================================================================
    # Loading the decoder creating the input vector
    # ================================================================================
    decoder, latent_dim = load_decoder(data_path=chosen_file)
    input_vec  = torch.nn.Parameter(torch.randn([1, latent_dim], device=decoder.device), requires_grad=True)
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logger = LoggerLatent(logdir=path)
    # ================================================================================
    # Creating the trainer object
    # ================================================================================
    trainer = TrainerLatent(input_vec=input_vec,
                            lr=LR,
                            mom=MOM,
                            beta=BETA_DKL,
                            sched_step=SCHEDULER_STEP,
                            sched_gamma=SCHEDULER_GAMMA,
                            grad_clip=GRAD_CLIP,
                            abs_sens=ABS_SENS,
                            sens_std=SENS_STD,
                            sens_mean=SENS_MEAN)
    # ================================================================================
    # Training
    # ================================================================================
    opt_vec = trainer.optimize_input(input_vec, decoder, 50000, logger, save_per_epoch=1)
    # ================================================================================
    # Plotting optimized latent space vector
    # ================================================================================
    plt.plot(opt_vec.cpu().detach().numpy().squeeze())
    plt.title('Optimized Latent Vector')
    plt.xlabel('index')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    phase = 1
    # ================================================================================
    # Training VAE on scatterer arrays and matching sensitivities
    # ================================================================================
    if phase == 1:
        c_path = None
        # c_path  = '..\\results\\20_12_2021_11_38'
        epoch = 160
        # enc_type = encoder_type_e.DENSE
        enc_type = encoder_type_e.VGG
        main_vae(enc_type, load_model=c_path, start_epoch=epoch)
    # ================================================================================
    # Using the decoder to maximize sensitivity prediction
    # ================================================================================
    if phase == 2:
        c_path = '..\\results\\5_12_2021_22_6'
        epoch = 20
        # main_optim_input(path=c_path, epoch=epoch)
        main_optim_latent(path=c_path, epoch=epoch)

