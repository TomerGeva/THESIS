# import numpy as np
# import os
import torch
from Config import *
from convolution_net.classses.Trainer import Trainer


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