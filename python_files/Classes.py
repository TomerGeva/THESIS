# import torch
# import numpy as np
from Config import *
import torch.nn as nn
import pandas   as pd
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from torch.autograd          import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data        import Dataset, DataLoader, random_split
from functions import *


# ============================================================
# defining the network
# ============================================================
class Net(nn.Module):
    # --------------------------------------------------------
    # input is a 2500X2500X1 signed matrix {-1, 1}
    # --------------------------------------------------------
    def __init__(self, device):
        super(Net, self).__init__()
        # a variable describing the forward process
        self.description = CONV_DESCRIPTION
        # computation location
        self.device      = device
        #                       in_channels | out_channels | kernel_size    | stride
        self.conv1 = nn.Conv2d(1            , FILTER_NUM[0], KERNEL_SIZE[0], STRIDES[0])
        self.conv2 = nn.Conv2d(FILTER_NUM[0], FILTER_NUM[1], KERNEL_SIZE[1], STRIDES[1])
        self.conv3 = nn.Conv2d(FILTER_NUM[1], FILTER_NUM[2], KERNEL_SIZE[2], STRIDES[2])
        self.conv4 = nn.Conv2d(FILTER_NUM[2], FILTER_NUM[3], KERNEL_SIZE[3], STRIDES[3])
        self.conv5 = nn.Conv2d(FILTER_NUM[3], FILTER_NUM[4], KERNEL_SIZE[4], STRIDES[4])
        self.conv6 = nn.Conv2d(FILTER_NUM[4], FILTER_NUM[5], KERNEL_SIZE[5], STRIDES[5])
        self.conv7 = nn.Conv2d(FILTER_NUM[5], FILTER_NUM[6], KERNEL_SIZE[6], STRIDES[6])
        # max-pooling to avoid over-fitting
        self.pool  = nn.MaxPool2d(MAX_POOL_SIZE)
        # performing affine operation: y = Wx + b
        x_dim, y_dim = self.compute_dim_sizes()
        self.fc1 = nn.Linear(FILTER_NUM[-1] * x_dim * y_dim, FC_LAYERS[0])
        self.fc2 = nn.Linear(FC_LAYERS[0], FC_LAYERS[1])
        self.fc3 = nn.Linear(FC_LAYERS[1], FC_LAYERS[2])
        self.fc4 = nn.Linear(FC_LAYERS[2], FC_LAYERS[3])

    def compute_dim_sizes(self):
        x_dim_size = xQuantize
        y_dim_size = yQuantize
        counter    = 0
        for action in range(len(self.description)):
            if self.description[action] == 'conv':
                x_dim_size = int((x_dim_size - (KERNEL_SIZE[counter] - STRIDES[counter])) / STRIDES[counter])
                y_dim_size = int((y_dim_size - (KERNEL_SIZE[counter] - STRIDES[counter])) / STRIDES[counter])
                counter += 1
            elif self.description[action] == 'pool':
                x_dim_size = int(x_dim_size / MAX_POOL_SIZE)
                y_dim_size = int(y_dim_size / MAX_POOL_SIZE)

        return x_dim_size, y_dim_size

    def forward(self, x):
        #                                               Convolutional section
        x = F.relu(self.conv1(x))                           # data_compression
        x = F.relu(self.conv2(x))                           # { first stage
        x = self.pool(F.relu(self.conv3(x)))                # }
        x = F.relu(self.conv4(x))                           # { second stage
        x = self.pool(F.relu(self.conv5(x)))                # }
        x = self.pool(F.relu(self.conv6(x)))                # third stage
        x = F.relu(self.conv7(x))                           # fourth stage
        x_dim, y_dim = self.compute_dim_sizes()       # Fully Connected section
        x = x.view(-1, FILTER_NUM[-1] * x_dim * y_dim)  # reshaping
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ============================================================
# defining the logger
# ============================================================
class Logger:
    def __init__(self, logdir):
        # -------------------------------------
        # tensorboard logger
        # -------------------------------------
        self.logger = SummaryWriter(logdir)
        self.logger_tag = []


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


class Trainer:
    def __init__(self, net, lr=1e-2, mu=0.9):
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.criterion = nn.MSELoss()
        # -------------------------------------
        # optimizer
        # -------------------------------------
        # self.optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mu)
        self.optimizer = optim.Adam(net.parameters(), lr=lr)
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

    def train(self, net, train_loader, test_loader, logger, save_per_epochs=1):
        """
        :param net: Net class object, which is the net we want to train
        :param train_loader: holds the training database
        :param test_loader: holds the testing database
        :param logger: logging the results
        :param save_per_epochs: flag indicating if you want to save
        :return: the function trains the network, and saves the trained network
        """
        losses = []
        accuracies_train = []
        accuracies_test = []
        print("Started training, learning rate: {}".format(self.learning_rate))
        # ----------------------------------------------
        # Begin of training
        # ----------------------------------------------
        net.eval()
        for epoch in range(self.epoch, EPOCH_NUM):
            train_loss = 0.0
            print("Starting Epoch #" + str(epoch + 1))
            for i, sample_batched in enumerate(train_loader):
                # ++++++++++++++++++++++++++++++++++++++
                # Extracting the grids and sensitivities
                # ++++++++++++++++++++++++++++++++++++++
                grids         = Variable(sample_batched['grid'].float()).to(net.device)
                sensitivities = Variable(sample_batched['sensitivity'].float()).to(net.device)

                # ++++++++++++++++++++++++++++++++++++++
                # Feed forward
                # ++++++++++++++++++++++++++++++++++++++
                self.optimizer.zero_grad()
                outputs = net(grids)

                # ++++++++++++++++++++++++++++++++++++++
                # Computing the loss
                # ++++++++++++++++++++++++++++++++++++++
                loss = self.criterion(outputs, sensitivities)

                # ++++++++++++++++++++++++++++++++++++++
                # Back propagation
                # ++++++++++++++++++++++++++++++++++++++
                loss.backward()
                self.optimizer.step()

                # ++++++++++++++++++++++++++++++++++++++
                # Documenting the loss
                # ++++++++++++++++++++++++++++++++++++++
                losses.append(loss.data.item())
                train_loss += loss.item() * grids.size(0)
            self.loss = train_loss / len(train_loader.dataset)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Testing accuracy at the end of the epoch
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            accuracies_train.append(accuracy_test(net, train_loader))
            accuracies_test.append(accuracy_test(net, test_loader))

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Documenting with tensorboard
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            logger.logger.add_scalars(logger.logger_tag + "_accuracy",
                                      {"Train_MSE_learning_rate_{}".format(self.learning_rate): accuracies_train[
                                          -1],
                                       "Test_MSE_learning_rate_{}".format(self.learning_rate): accuracies_test[
                                           -1]},
                                      epoch + 1)
            logger.logger.add_scalars(logger.logger_tag + "_loss",
                                      {"learning_rate_{}".format(self.learning_rate): self.loss},
                                      epoch + 1)

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Saving the training state
            # save every x epochs and on the last epoch
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # Printing log to screen
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            print("Epoch: {}/{} \tTraining loss: {:.10f} \tTrain MSE: {:.6f} \tTest MSE: {:.6f}".format(
                epoch + 1, EPOCH_NUM, self.loss,
                accuracies_train[-1], accuracies_test[-1]))


# ============================================================
# defining the dataset class
# ============================================================

class ScattererCoordinateDataset(Dataset):
    """
    This is a dataset used to store the coordinate array of the scatterers, and the corresponding sensitivity.
    A sample of the dataset will be a dictionary {'grid': 2D array, 'sensitivity': target sensitivity}
    Out dataset will take additional argument 'transform' so that any required processing can be applied on the sample.
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
        :param csv_file: path to the file with all the database
        :param transform: transformation flag of the data
        """
        self.csv_data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # ---------------------------------------------
        # extracting the sensitivity
        # ---------------------------------------------
        sensitivity = self.csv_data.iloc[idx, 0] / NORM_FACT

        # ---------------------------------------------
        # extracting the points
        # ---------------------------------------------
        points = self.csv_data.iloc[idx, 1:]
        points = np.array([points])
        points = points.astype('float').reshape(-1, 2)

        # ---------------------------------------------
        # converting points from micro meter to pixels
        # ---------------------------------------------
        pixel_points = micrometer2pixel(points)

        # ---------------------------------------------
        # converting the points to a 2-D array
        # ---------------------------------------------
        grid_array = points2mat(pixel_points)

        # -------------------------------------------
        # creating the sample dict
        # -------------------------------------------
        sample = {'grid': grid_array,
                  'sensitivity': sensitivity}

        # -------------------------------------------
        # transforming sample if given
        # -------------------------------------------
        if self.transform:
            sample = self.transform(sample)

        return sample


# ============================================================
# defining transform class, converting 2D arrays to tensors
# ============================================================
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grid, sensitivity = sample['grid'], sample['sensitivity']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # in this case there is only one channel, C = 1, thus we use expand_dims instead of transpose
        grid        = np.expand_dims(grid, axis=0)
        sensitivity = np.expand_dims(sensitivity, axis=0)
        return {'grid': torch.from_numpy(grid),
                'sensitivity': torch.from_numpy(np.array(sensitivity))}


# ============================================================
# defining function which manipulate the classes above
# ============================================================
def import_data_sets(batch_size, test_rate):
    """
    This function imports the train and test database
    :param batch_size: size of each batch in the databases
    :param test_rate: percentage of the total dataset which will be dedicated to taining
    :return: two datasets, training and test
    """
    # --------------------------------------------------------
    # Importing complete dataset
    # --------------------------------------------------------
    data = ScattererCoordinateDataset(csv_file=path_database, transform=ToTensor())

    # --------------------------------------------------------
    # Computing the lengths
    # --------------------------------------------------------
    length    = round(len(data))
    train_len = round(length * (1 - test_rate))
    test_len  = length - train_len

    # --------------------------------------------------------
    # Splitting randomly into two sets
    # --------------------------------------------------------
    train_data, test_data = random_split(data, [train_len, test_len])

    # --------------------------------------------------------
    # Creating the loaders
    # --------------------------------------------------------
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

# if __name__ == '__main__':
#     data = ScattererCoordinateDataset(csv_file=path_database, transform=ToTensor())
#
#     sample = data[5]
#     print('{} \t{} \t{} \t {}'.format(5, sample['grid'].size(), sample['sensitivity'], sample['sensitivity'].size()))