from config_omega import *
import torch
from database_functions import ModelManipulationFunctions
from classes.LoggerOmega import LoggerOmega
from classes.TrainerOmega import TrainerOmega
from classes.Model_Omega import OmegaModel
from auxiliary_functions import _init_
from PCurrentDataset import import_pcurrents_dataset


def main():
    mmf = ModelManipulationFunctions()
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logdir = _init_(PATH_LOGS)
    logger = LoggerOmega(logdir=logdir)
    logger.result_space = 22.19
    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ================================================================================
    # Importing the database
    # ================================================================================
    train_loader, test_loader = import_pcurrents_dataset(BATCH_SIZE, PATH_DATABASE_TRAIN, PATH_DATABASE_TEST,
                                                         omega_factor=OMEGA_FACTOR,
                                                         shot_noise=SHOT_NOISE,
                                                         sampling_rate=SAMPLING_RATE,
                                                         num_workers=NUM_WORKERS)
    # ================================================================================
    # Creating the model
    # ================================================================================
    model = OmegaModel(device, FC_TOPOLOGY)
    mmf.initialize_weights(model, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD, method='xavier')
    model.to(device)
    # ================================================================================
    # Creating the trainer
    # ================================================================================
    trainer = TrainerOmega(model,
                           num_epochs=EPOCH_NUM,
                           lr=LR, mom=MOM,
                           sched_step=SCHEDULER_STEP, sched_gamma=SCHEDULER_GAMMA,
                           grad_clip=GRAD_CLIP,
                           omega_factor=OMEGA_FACTOR, shot_noise=SHOT_NOISE, sampling_rate=SAMPLING_RATE)
    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(model, logger, train_loader, test_loader, save_per_epochs=SAVE_PER_EPOCH)


if __name__ == '__main__':
    main()
