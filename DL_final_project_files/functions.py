from Config import *
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


# **********************************************************************************************************************
# THIS SECTION DEALS WITH FUNCTIONS REGARDING WITH THE DATABASES
# **********************************************************************************************************************
def import_data_sets(train_data_path, val_data_path):
    """
    :return: this function returns the train database and validation databases
    """
    # -------------------------------------------------------------------
    # downloading the relevant datasets
    # -------------------------------------------------------------------
    with open(train_data_path, 'r') as train_file:
        train_data = train_file.read()

    with open(val_data_path, 'r') as val_file:
        val_data = val_file.read()

    return train_data, val_data


# **********************************************************************************************************************
# THIS SECTION DEALS WITH FUNCTIONS REGARDING WITH THE NETS
# **********************************************************************************************************************
def accuracy_test(epoch, net, loader):
    correct = 0.0
    total = 0.0
    i = epoch * BATCH_SIZE
    with torch.no_grad():
        for inputs, symbols in loader:
            inputs  = Variable(inputs.float()).to(net.device)
            symbols = Variable(symbols).long().to(net.device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += symbols.size(0)
            correct += (predicted == symbols).sum()
            if i == (epoch+1) * BATCH_SIZE:
                break
    return (100 * correct / total).item()


def save_state_train(trainer, filename, net, epoch, lr, loss):
    """Saving model and optimizer to drive, as well as current epoch and loss
    # When saving a general checkpoint, to be used for either inference or resuming training, you must save more
    # than just the model’s state_dict.
    # It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are
    # updated as the model trains.
    """
    path = os.path.join(PATH_MODELS, filename)
    data_to_save = {'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': loss,
                    'lr': lr
                    }
    torch.save(data_to_save, path)


# **********************************************************************************************************************
# AUXILLARY FUNCTIONS
# **********************************************************************************************************************
def create_logger_tag(lr, mom):
    return "learning_rate_" + str(lr) + "_momentum_" + str(mom)

# if __name__ == "__main__":
#     train_loader, test_loader = import_data_sets(BATCH_SIZE)
#     show_batch(train_loader, BATCH_SIZE, labels_dict)
