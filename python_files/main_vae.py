# import numpy as np
# import os
import torch
import os
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
from database_functions         import load_state_train


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
                         encoder_topology=VGG_ENCODER_TOPOLOGY,
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


def main_optim_input(path=None, epoch=None):
    # ================================================================================
    # creating full file path
    # ================================================================================
    save_files = [os.path.join(path, d) for d in os.listdir(path) if "epoch" in d]
    if epoch is None:
        epoch_nums = [int(file.split(sep='_')[-1][0:-4]) for file in save_files[1:]]
        epoch = max(epoch_nums)
    chosen_file = [d for d in save_files if str(epoch) in d][0]
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
    trainer.optimize_input(input_mat, mod_vae, 2000, logger, save_per_epoch=1)


def main_optim_latent(path=None, epoch=None):
    # ================================================================================
    # creating full file path
    # ================================================================================
    save_files = [os.path.join(path, d) for d in os.listdir(path) if "epoch" in d]
    if epoch is None:
        epoch_nums = [int(file.split(sep='_')[-1][0:-4]) for file in save_files[1:]]
        epoch = max(epoch_nums)
    chosen_file = [d for d in save_files if str(epoch) in d][0]

    # ================================================================================
    # Loading the decoder creating the input vector TODO: need to set a path to a requested model folder
    # ================================================================================
    decoder, latent_dim = load_decoder(data_path=path)
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
        c_path = '..\\results\\18_11_2021_8_21'
        epoch = 12
        main_optim_input(path=c_path, epoch=epoch)
