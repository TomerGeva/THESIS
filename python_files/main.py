# import numpy as np
# import os
import torch
from Config import *
from Logger import Logger
from Trainer import Trainer
from Net_dynamic import ConvNet
from functions import initialize_weights
from ScatterCoordinateDataset import import_data_sets


def main_old():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # location of the computation: CPU or GPU
    if gather_DB:
        # =====================================
        # gathering the database and saving
        # =====================================
        database = gather_data(path)
        torch.save(database, SAVE_PATH_DB)
    else:
        database = torch.load(SAVE_PATH_DB)

    net = Net()
    net.to(device)  # allocating the computation to the CPU or GPU

    if train:
        trainer = Trainer(net)
        trainer.train(net, database)


def main():
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
    main()




