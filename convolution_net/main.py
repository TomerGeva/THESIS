# import numpy as np
# import os
import torch
from convolution_net.Config import *
from convolution_net.classses.Logger import Logger, LoggerNew
from convolution_net.classses.Trainer import Trainer, TrainerVAE
from convolution_net.classses.ConvNet import ConvNet
from functions                  import initialize_weights
from common.database_classes.ScatterCoordinateDataset import import_data_sets


def main_conv():
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logger = Logger(PATH_LOGS)
    logger.logger_tag = 'Training_raw'

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
    net = ConvNet(device)
    initialize_weights(net, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD)
    net.to(device)  # allocating the computation to the CPU or GPU
    trainer = Trainer(net, lr=LR, mu=MU)
    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(net, train_loader, test_loader, logger)


if __name__ == '__main__':
    main_conv()
    # train_loader, test_loader = import_data_sets(BATCH_SIZE, 0.15)
    # for sample in test_loader:
    #     print(torch.max(sample['sensitivity']))





