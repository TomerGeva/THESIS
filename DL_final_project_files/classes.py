from Config import *
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
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
# defining symbol sequence class, dealing with sequence popping
# ==============================================================
class SymbolDatasetIterable(IterableDataset):
    def __init__(self, constellation, precursors=5, postcursors=20, train=True):
        """
        :param constellation: possible symbol values!
        :param precursors: number of precursors used in the input layer
        :param postcursors: number of postcursors used in the input layer
        :param train: if true, the system loads the train database, otherwise loads the validation database
        """
        # --------------------------------------------------------------------------------------
        # Reading the data file from the logdir
        # --------------------------------------------------------------------------------------
        if train:
            self.csv_data, _ = import_data_sets(PATH_TRAIN_DATA, PATH_VAL_DATA)
        else:
            _, self.csv_data = import_data_sets(PATH_TRAIN_DATA, PATH_VAL_DATA)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Splitting the stringed data to arrays of inputs and outputs
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        database = self.csv_data.split(sep='\n')
        database = [[float(jj) for jj in database[ii].split(',')]for ii in range(1, len(database)-1)]
        database = np.array(database)

        # --------------------------------------------------------------------------------------
        # Set creates an un-indexed and unorganized unique values of the words in the text.
        # This set is then turned into a list
        # --------------------------------------------------------------------------------------
        self.constellation = constellation

        # --------------------------------------------------------------------------------------
        # Creating the input sequences
        #   1. each sequence is (<precursors> + <postcursors> + 1) symbols long
        #   2. the corresponding output is the main cursor, i.e., the (<precursors> + 1) symbol
        #   3. expanding the dimension for pytorch functionality purposes
        # --------------------------------------------------------------------------------------
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Extracting the data
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        seq_len = precursors + postcursors + 1
        inputs  = database[:, 1]
        outputs = database[:, 0]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # converting symbols to labels
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        outputs = np.array([SYMBOLS2LABELS_DICT[ii] for ii in outputs])
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # dividing into sequences
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.inputs_per_iter  = [inputs[ii:ii+seq_len] for ii in range(len(inputs) - postcursors)]
        self.outputs_per_iter = outputs[postcursors:]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # expanding input dimension
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.inputs_per_iter  = [np.expand_dims(ii, axis=0) for ii in self.inputs_per_iter]
        # self.outputs_per_iter = [np.expand_dims(ii, axis=0) for ii in self.outputs_per_iter]

    def __len__(self):
        # --------------------------------------------------------------------------------------
        # the length of the database is the number of sequences
        # --------------------------------------------------------------------------------------
        return len(self.inputs_per_iter)

    def __iter__(self):
        return iter(tuple(zip(self.inputs_per_iter, self.outputs_per_iter)))


# ==============================================================
# this class is the net which will reverse the channel response
# ==============================================================
"""
class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.device      = device
        self.description = LAYER_DESCRIPTION
        # ---------------------------------------------------------
        # constructing the first layer
        # ---------------------------------------------------------
        self.layer1 = nn.Sequential(nn.Conv1d(1, FILTER_NUM[0], kernel_size=KERNEL_SIZE[0], padding=PADDING[0]),
                                    nn.BatchNorm1d(FILTER_NUM[0]),
                                    nn.ReLU(),
                                    nn.MaxPool1d(MAX_POOL_SIZE))

        # ---------------------------------------------------------
        # constructing the second layer, as a function of the mode
        # ---------------------------------------------------------
        self.layer2 = nn.Sequential(nn.Conv1d(FILTER_NUM[0], FILTER_NUM[1],
                                              kernel_size=KERNEL_SIZE[1], padding=PADDING[1]),
                                    nn.BatchNorm1d(FILTER_NUM[1]),
                                    nn.ReLU(),
                                    nn.MaxPool1d(MAX_POOL_SIZE))

        # ---------------------------------------------------------
        # computing the dimensions of the convolution output
        # ---------------------------------------------------------
        x_dim = self.compute_dim_sizes()

        # ---------------------------------------------------------
        # Fully Connected Layers
        # ---------------------------------------------------------
        self.fc1 = nn.Linear(x_dim * FILTER_NUM[-1], FC_LAYERS[0])
        self.fc2 = nn.Linear(FC_LAYERS[0], FC_LAYERS[1])

    def compute_dim_sizes(self):
        x_dim_size = X_SIZE
        counter    = 0
        for action in range(len(self.description)):
            if self.description[action] == 'conv':
                x_dim_size = int((x_dim_size - (KERNEL_SIZE[counter] - STRIDES[counter]) + 2*PADDING[counter])
                                 / STRIDES[counter])
                counter += 1
            elif self.description[action] == 'pool':
                x_dim_size = int(x_dim_size / MAX_POOL_SIZE)

        return x_dim_size

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.device      = device
        self.description = LAYER_DESCRIPTION
        self.conv_len    = len(FILTER_NUM) - 1  # THS INPUT IS NOT INCLUDED, THUS THE REDUCTION OF 1
        self.fc_len      = len(FC_LAYERS)
        self.layers      = nn.ModuleList()
        # ---------------------------------------------------------
        # Creating the convolution blocks
        # ---------------------------------------------------------
        for ii in range(self.conv_len):
            self.layers.append(self._conv_block(FILTER_NUM[ii],
                                                FILTER_NUM[ii+1],
                                                KERNEL_SIZE[ii],
                                                STRIDES[ii],
                                                PADDING[ii],
                                                MAX_POOL_SIZE[ii])
                               )
        # ---------------------------------------------------------
        # computing the dimensions of the convolution output
        # ---------------------------------------------------------
        x_dim = self.compute_dim_sizes()
        # ---------------------------------------------------------
        # Fully Connected Layers
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            if ii == 0:
                self.layers.append(self._fc_block(x_dim * FILTER_NUM[-1], FC_LAYERS[ii], activation=True))
            elif ii == (self.fc_len - 1):
                self.layers.append(self._fc_block(FC_LAYERS[ii-1], FC_LAYERS[ii], activation=False))
            else:
                self.layers.append(self._fc_block(FC_LAYERS[ii - 1], FC_LAYERS[ii], activation=True))

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding, maxpool=1):
        return nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False,
                      ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(maxpool)
        )

    def _fc_block(self, in_size, out_size, activation=True):
        if activation:
            return nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU()
            )
        else:
            return nn.Linear(in_size, out_size)

    def compute_dim_sizes(self):
        x_dim_size = X_SIZE
        counter = 0
        for action in range(len(self.description)):
            if self.description[action] == 'conv':
                x_dim_size = int((x_dim_size - (KERNEL_SIZE[counter] - STRIDES[counter]) + 2 * PADDING[counter])
                                 / STRIDES[counter])
            elif self.description[action] == 'pool':
                x_dim_size = int(x_dim_size / MAX_POOL_SIZE[counter])
                counter += 1

        return x_dim_size

    def forward(self, x):
        # ---------------------------------------------------------
        # passing through the convolution blocks
        # ---------------------------------------------------------
        for ii in range(self.conv_len):
            layer = self.layers[ii]
            x = layer(x)
        # ---------------------------------------------------------
        # flattening for the FC layers
        # ---------------------------------------------------------
        x = x.view(x.size(0), -1)
        # ---------------------------------------------------------
        # passing through the fully connected blocks
        # ---------------------------------------------------------
        for ii in range(self.fc_len):
            layer = self.layers[ii + self.conv_len]
            x = layer(x)

        return x


# ==============================================================
# logger for logging the training progress
# ==============================================================
class Logger:
    def __init__(self, logdir):
        # -------------------------------------
        # tensorboard logger
        # -------------------------------------
        self.logger = SummaryWriter(logdir)
        self.logger_tag = []


# ==============================================================
# trainer object which will be used in the training
# ==============================================================
class Trainer:
    def __init__(self, net, lr=0.01, mom=0.9):
        # -------------------------------------
        # cost function
        # -------------------------------------
        self.criterion = nn.CrossEntropyLoss()
        # -------------------------------------
        # optimizer
        # -------------------------------------
        self.optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom)
        # -------------------------------------
        # Initializing the start epoch to zero
        # if not None, the model is pre-trained
        # -------------------------------------
        self.start_epoch = 0

        # -------------------------------------
        # Misc training parameters
        # -------------------------------------
        self.loss          = []
        self.learning_rate = lr
        self.momentum      = mom

    def train(self, net, train_loader, test_loader, logger, save_per_epochs=1):
        """
        :param net: Net class object, which is the net we want to train
        :param train_loader: holds the training database
        :param test_loader: holds the testing database
        :param logger: logging the results
        :param save_per_epochs: flag indicating if you want to save
        :return: the function trains the network, and saves the trained network
        """
        losses           = []
        accuracies_train = []
        accuracies_test  = []
        print("Started training, learning rate: {}".format(self.learning_rate))
        # ----------------------------------------------------------
        # drop-out and batch normalization  behave differently in
        # training and evaluation, thus we use the following:
        # ----------------------------------------------------------
        net.train()
        for epoch in range(self.start_epoch, EPOCH_NUM):
            train_loss = 0.0
            for i, (inputs, symbols) in enumerate(train_loader):
                # ++++++++++++++++++++++++++++++++++++++
                # Extracting the images and labels
                # ++++++++++++++++++++++++++++++++++++++
                inputs = Variable(inputs.float()).to(net.device)
                symbols = Variable(symbols).long().to(net.device)

                # ++++++++++++++++++++++++++++++++++++++
                # Feed forward
                # ++++++++++++++++++++++++++++++++++++++
                self.optimizer.zero_grad()
                outputs = net(inputs)

                # ++++++++++++++++++++++++++++++++++++++
                # Computing the loss
                # ++++++++++++++++++++++++++++++++++++++
                loss = self.criterion(outputs, symbols)

                # ++++++++++++++++++++++++++++++++++++++
                # Back propagation
                # ++++++++++++++++++++++++++++++++++++++
                loss.backward()
                self.optimizer.step()

                # ++++++++++++++++++++++++++++++++++++++
                # Documenting the loss
                # ++++++++++++++++++++++++++++++++++++++
                losses.append(loss.data.item())
                train_loss += loss.item() * inputs.size(0)

            self.loss = train_loss / len(train_loader.dataset)
            # ==========================================
            # Testing accuracy at the end of the epoch
            # ==========================================
            net.eval()
            accuracies_train.append(accuracy_test(epoch, net, train_loader))
            accuracies_test.append(accuracy_test(epoch, net, test_loader))
            net.train()

            # ==========================================
            # Documenting with tensorboard
            # ==========================================
            logger.logger.add_scalars(logger.logger_tag + "_accuracy",
                                    {"Train_accuracy_learning_rate_{}".format(self.learning_rate): accuracies_train[-1],
                                     "Test_accuracy_learning_rate_{}".format(self.learning_rate): accuracies_test[-1]},
                                    epoch + 1)
            logger.logger.add_scalars(logger.logger_tag + "_loss",
                                    {"learning_rate_{}".format(self.learning_rate): self.loss},
                                    epoch + 1)

            # ==========================================
            # Saving the training state
            # save every x epochs and on the last epoch
            # ==========================================
            if epoch % save_per_epochs == 1 or epoch + 1 == EPOCH_NUM:
                save_state_train(self, filename=os.path.join("{}".format(logger.logger_tag),
                                                             "lr_{}_epoch_{}.tar".format(self.learning_rate, epoch+1)),
                                 net=net, epoch=epoch+1, lr=self.learning_rate, loss=self.loss)

            # ==========================================
            # Printing log to screen
            # ==========================================
            print("Epoch: {}/{} \tTraining loss: {:.6f} \tTrain accuracy: {:.6f}% \tTest accuracy: {:.6f}%".format(
                epoch + 1, EPOCH_NUM, self.loss,
                accuracies_train[-1], accuracies_test[-1]))


# ============================================================
# defining function which manipulate the classes above
# ============================================================
def import_data_loaders(batch_size):
    """
    This function imports the train and test database
    :param batch_size: size of each batch in the databases
    :param sequence_len: length of each sequence in the database
    :return: two datasets, training and validation
    """
    # --------------------------------------------------------
    # Importing complete dataset
    # --------------------------------------------------------
    train_database = SymbolDatasetIterable(CONSTELLATION, PRECURSOR_NUM, POSTCURSOR_NUM, train=True)
    valid_database = SymbolDatasetIterable(CONSTELLATION, PRECURSOR_NUM, POSTCURSOR_NUM, train=False)

    # --------------------------------------------------------
    # Creating the loaders
    # --------------------------------------------------------
    train_loader = DataLoader(train_database, batch_size=batch_size, drop_last=True, num_workers=1)
    test_loader  = DataLoader(valid_database, batch_size=batch_size, drop_last=True, num_workers=1)

    return train_loader, test_loader


def load_state_train(mode):
    """Loads training state from drive or memory
    when loading the dictionary, the function also arranges the data in such a manner which allows to
    continure the training
    """
    # -------------------------------------
    # assembling the logdir
    # -------------------------------------
    filename = r"lr_0.01_epoch_50.tar"
    path = os.path.join(path_models, modes_dict[mode], filename)

    # -------------------------------------
    # allocating device type
    # -------------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------
    # loading the dictionary
    # -------------------------------------
    checkpoint = torch.load(path, map_location=device)

    # -------------------------------------
    # arranging the data
    # -------------------------------------
    net     = Net(mode, device)
    net.to(device)  # allocating the computation to the CPU or GPU
    trainer = Trainer(net)

    trainer.loss          = checkpoint['loss']
    trainer.learning_rate = checkpoint['lr']
    trainer.start_epoch   = checkpoint['epoch']

    net.load_state_dict(checkpoint['model_state_dict'])
    if mode == WEIGHT_DECAY:
        trainer.optimizer = optim.SGD(net.parameters(), lr=trainer.learning_rate, weight_decay=0.01)
    else:
        trainer.optimizer = optim.SGD(net.parameters(), lr=trainer.learning_rate)
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return net, trainer


def train_new_net(logger, train_loader, test_loader, device=None):
    # ----------------------------------------------------------------------------
    # Test Parameters
    # ----------------------------------------------------------------------------
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Starting training for the new network")
    # ----------------------------------------------------------------------------
    # Creating the net and allocating computation device
    # ----------------------------------------------------------------------------
    net = Net(device)
    net.to(device)  # allocating the computation to the CPU or GPU

    # ----------------------------------------------------------------------------
    # Creating the trainer object, and adding the logger tag
    # ----------------------------------------------------------------------------
    trainer = Trainer(net, MU)
    logger.logger_tag = create_logger_tag(trainer.learning_rate, trainer.momentum)

    # ----------------------------------------------------------------------------
    # Initialize training
    # ----------------------------------------------------------------------------
    trainer.train(net, train_loader, test_loader, logger)


# if __name__ == "__main__":
    # iterable = SymbolDatasetIterable(CONSTELLATION)
    # for batch in iterable:
    #     print(batch)
    # train_loader, test_loader = import_data_loaders(128)
    # for ii, batch in enumerate(train_loader):
    #     print(ii)
#     print("hi")
