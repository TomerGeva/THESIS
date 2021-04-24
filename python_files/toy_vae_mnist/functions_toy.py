from Config import *
import torch
import torch.nn as nn
import torch.nn.functional      as F
import matplotlib.pyplot        as plt
import torchvision.transforms   as transforms
from time               import sleep
from torch.autograd     import Variable
from torchvision        import datasets
from torch.utils.data   import DataLoader


def import_data_sets(batch_size, x_size=28, y_size=28):
    """
    :param batch_size: the size of each batch to run through the network
    :param x_size: size of the first dimension
    :param y_size: size of the second dimension
    :return: this function returns the train database and test database, after organizing it as loaders
    """
    # -------------------------------------------------------------------
    # Transformations - defining useful transformations
    # -------------------------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.Resize((x_size, y_size)),
            transforms.ToTensor(),
            # transforms.Normalize([TRANSFORM_NORM] * IMG_CHANNELS, [TRANSFORM_NORM] * IMG_CHANNELS)
        ])

    # -------------------------------------------------------------------
    # downloading the relevant datasets
    # -------------------------------------------------------------------
    train_data = datasets.FashionMNIST(root='data', download=True, train=True, transform=transform)
    test_data  = datasets.FashionMNIST(root='data', download=True, train=False, transform=transform)

    # -------------------------------------------------------------------
    # preparing the data loaders
    # -------------------------------------------------------------------
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def show_batch(train_loader, batch_size, labels_dict):
    """
    :param train_loader: the loader of the train database, showing examples
    :param batch_size: the size of a batch
    :param labels_dict: the label dictionary
    :return: this function plots the batch, for us to see the database
    """
    data_iteration = iter(train_loader)
    images, labels = data_iteration.next()
    images = images.numpy()

    fig = plt.figure(figsize=(10, 10))
    row_num = 8
    for ii in range(batch_size):
        # Start next subplot.
        plt.subplot(row_num, batch_size / row_num, ii + 1, title=labels_dict[(labels[ii].item())])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(images[ii]), cmap=plt.cm.binary)

    plt.show()


def display_reconstruction(batch, batch_size):
    """
    :param batch: a batch of images
    :param batch_size: the size of a batch
    :return: this function plots the batch, for us to see the database
    """

    images = batch.numpy()

    fig = plt.figure(figsize=(10, 10))
    row_num = 8
    for ii in range(batch_size):
        # Start next subplot.
        plt.subplot(row_num, batch_size / row_num, ii + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(images[ii]), cmap=plt.cm.binary)

    plt.ioff()
    plt.show()


def cost_function(pictures, reconstruction, mu, std):
    # ================================================
    # Computing the BCE loss
    # ================================================
    bce_loss = F.binary_cross_entropy(pictures, reconstruction.detach(), reduction='mean')
    # computing D_kl means summing over all dimensions, and averaging over the batch
    d_kl     = torch.mean(0.5 * torch.sum(std.pow(2) + mu.pow(2) - 1 - 2*std.log(), dim=1))

    return bce_loss + BETA * d_kl


def train(vae, optimizer, train_loader, test_loader, device):
    print("Started training")
    vae.train()
    # ================================================
    # Going over ?? epochs
    # ================================================
    for epoch in range(100):
        train_cost = 0.0
        # ----------------------------------------
        # All batches per epoch
        # ----------------------------------------
        for pictures, _ in train_loader:
            pictures = Variable(pictures.to(device), requires_grad=True)
            # ___________Forward pass_____________
            reconstruction, mu, std = vae(pictures)
            cost = cost_function(pictures, reconstruction, mu, std)
            train_cost += cost.item()
            # ___________Backward calc____________
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        # ----------------------------------------
        # Printing  progress
        # ----------------------------------------
        print('Epoch ', epoch, ' total cost: ', train_cost)
        display_reconstruction(reconstruction.detach(), 32)
