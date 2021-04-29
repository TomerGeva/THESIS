from Config import *
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import pandas   as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import IterableDataset, DataLoader, random_split
from functions import *


# ==============================================================
# Defining the database class, dealing with all the data
# ==============================================================
class ScattererCoordinateDataset(IterableDataset):
    """
        This is a dataset used to store the coordinate array of the scatterers, and the corresponding sensitivity.
        * A sample of the dataset will be a tuple ('2D-grid', 'sensitivity')
            ** the database will be an iterable
        * Optional future feature: Out dataset will take additional argument 'transform' so that any required processing
          can be applied on the sample.
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
        :param csv_file: logdir to the file with all the database
        :param transform: transformation flag of the data
        """
        self.transform = transform
        # --------------------------------------------------
        # Reading the database csv file
        # --------------------------------------------------
        csv_data = pd.read_csv(csv_file)
        # --------------------------------------------------
        # Extracting the sensitivities
        # --------------------------------------------------
        sensitivities      = csv_data.iloc[:, 0] / NORM_FACT
        self.sensitivities = sensitivities.values.tolist()
        # --------------------------------------------------
        # Extracting the coordinations as pixel locations
        # --------------------------------------------------
        self.grids_array = []
        for ii in range(len(csv_data)):
            points = csv_data.iloc[ii, 1:]
            points = np.array([points])
            points = points.astype('float').reshape(-1, 2)
            # ++++++++++++++++++++++++++++++++++++++++++
            # Converting from micrometer to pixel
            # ++++++++++++++++++++++++++++++++++++++++++
            pixel_points = micrometer2pixel(points)
            # ++++++++++++++++++++++++++++++++++++++++++
            # Converting from pixel to actual grid
            # ++++++++++++++++++++++++++++++++++++++++++
            grid_array = points2mat(pixel_points)
            self.grids_array.append(grid_array)
        # --------------------------------------------------
        # Expanding dimensions as a pre-requisite of pytorch
        # --------------------------------------------------
        self.grids_array = [np.expand_dims(grid, axis=0) for grid in self.grids_array]

    def __len__(self):
        return len(self.sensitivities)

    def __iter__(self):
        return iter(tuple(zip(self.grids_array, self.sensitivities)))


# ============================================================
# defining the trainer
# ============================================================
class TrainerOld:
    def __init__(self, net, lr=1e-2, mu=0.9):
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.criterion = nn.MSELoss()
        # -------------------------------------
        # optimizer
        # -------------------------------------
        self.optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mu)
        # -------------------------------------
        # Initializing the start epoch to zero
        # if not zero, the model is pre-trained
        # -------------------------------------
        self.epoch = 0
        # -------------------------------------
        # Misc training parameters
        # -------------------------------------
        self.loss          = []
        self.learning_rate = lr
        self.mu            = mu

    def train(self, net, database, logger, save_per_epoch=1):
        """
        :param net: Net class object, which is the net we want to train
        :param database: training database. This is a dictionary which hold 3 sub dictionaries, and the database size:
        [scat_arrays] - contains the grid coordinates of the scatterers. Each array hold a set of scatterer locations
        [sensitivity] - contains the matching sensitivity of the respective array of scatterers
        [scat_num]    - contains the number of scatterer that produces the maximal sensitivity for the respective array
        [size]        - hold the size of the database
        :param logger: a Logger class, responsible for logging the progress
        :param save_per_epoch: number of epochs between each save. the default is saving after every epoch
        :return: the function trains the network, and saves the trained network
        """
        losses = []
        accuracies_train = []
        accuracies_test = []
        # ==================================================
        # breaking the database into the components
        # ==================================================
        arrays_dict              = database['scat_arrays']
        target_sensitivity_dict  = database['sensitivity']
        mini_batch_input         = torch.zeros([BATCH_SIZE, 1, xQuantize, yQuantize])
        mini_batch_target_output = torch.zeros([BATCH_SIZE, 1])

        print("Started training, learning rate: {}".format(self.learning_rate))
        # ==================================================
        # Training
        # ==================================================
        net.eval()
        for epoch in range(self.epoch, EPOCH_NUM):
            counter      = 1
            mini_counter = 1
            running_loss = 0.0
            for key in arrays_dict.keys():
                # ------------------------------------------
                # getting input and target output
                # ------------------------------------------
                arr        = arrays_dict[key]
                input_grid = array2mat(arr)
                target_sen = torch.abs(torch.tensor([[target_sensitivity_dict[key] * NORM_FACT]]))

                # ------------------------------------------
                # forward passing through the network
                # ------------------------------------------
                if mini_counter % BATCH_SIZE == 0:
                    print('Epoch number ', str(epoch + 1), ', batch number ', str(counter))
                    mini_counter = 1  # resetting count
                    net_output_sens  = net(mini_batch_input)

                    # ++++++++++++++++++++++++++++++++++++++
                    # back-update operations
                    # ++++++++++++++++++++++++++++++++++++++
                    loss = self.criterion(net_output_sens, mini_batch_target_output)  # defining the loss
                    net.zero_grad()                                                   # zero the net gradients
                    loss.backward()                                                   # back-propagating
                    self.optimizer.step()                                             # updating the weights

                    running_loss += loss.item()

                    if counter % 10 == 0:  # print every N mini-batches
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, counter, running_loss / 10))
                        running_loss = 0.0

                    counter += 1
                else:
                    mini_batch_input[mini_counter-1, 0, :, :]   = input_grid
                    mini_batch_target_output[mini_counter-1, 0] = target_sen
                    mini_counter += 1

        print('Finished Training')
        torch.save(net.state_dict(), SAVE_PATH_NET)


if __name__ == '__main__':
    database = ScattererCoordinateDataset(csv_file=path_database)
    print('hi')
