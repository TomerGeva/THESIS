# import numpy as np
# import os
# import torch
from Config import *
from Classes import *


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
    logger = Logger(path_logs)
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
    net = Net(device)
    net.to(device)  # allocating the computation to the CPU or GPU
    trainer = Trainer(net, lr=LR, mu=MU)
    trainer.train(net, train_loader, test_loader, logger)

    """
    for i_batch, sample_batched in enumerate(train_loader):
        print('{} \t{} \t{} \t {}'.format(i_batch, sample_batched['grid'].size(),
                                          sample_batched['sensitivity'], sample_batched['sensitivity'].size()))
        if i_batch == 3:
            break
    print('hi')
    """


if __name__ == '__main__':
    main()




