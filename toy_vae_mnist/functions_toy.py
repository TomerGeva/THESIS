from convolution_net.Config import *
import torch
import torch.nn as nn
import torch.nn.functional      as F
import matplotlib.pyplot        as plt
import torchvision.transforms   as transforms
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
            transforms.Normalize([TRANSFORM_NORM] * IMG_CHANNELS, [TRANSFORM_NORM] * IMG_CHANNELS)
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
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
    for ii in range(min(images.shape[0], 32)):
        # Start next subplot.
        plt.subplot(row_num, int(32 / row_num), ii + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(images[ii]), cmap=plt.cm.binary)

    plt.ioff()
    plt.show()


def initialize_weights_toy(net, mean, std):
    """
    :param net: the model which is being normalized
    :param mean: the target mean of the weights
    :param std: the target standard deviation of the weights
    :return: nothing, just adjusts the weights
    """
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Linear)):
            nn.init.normal_(module.weight.data, mean, std)
            module.bias.data.zero_()


def cost_function(pictures, reconstruction, mu, logvar):
    # ================================================
    # Computing the BCE loss
    # ================================================
    mse_loss = F.mse_loss(pictures, reconstruction, reduction='sum')
    # computing D_kl means summing over all dimensions, and averaging over the batch
    d_kl     = torch.sum(0.5 * torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar, dim=1))
    return mse_loss, d_kl, mse_loss + BETA * d_kl


def train(vae, optimizer, train_loader, test_loader, device):
    print("Started training")
    vae.train()
    # ================================================
    # Going over ?? epochs
    # ================================================
    for epoch in range(1000):
        train_cost  = 0.0
        counter     = 0
        # ----------------------------------------
        # All batches per epoch
        # ----------------------------------------
        for pictures, _ in test_loader:
            if counter >= 1:
                break
            pictures = Variable(pictures.to(device))
            # ___________Forward pass_____________
            reconstruction, mu, logvar = vae(pictures)
            _, _, cost = cost_function(pictures, reconstruction, mu, logvar)
            train_cost += cost.item()
            # ___________Backward calc____________
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            counter += 1
            if False:
                plt.subplot(1, 2, 1)
                plt.imshow(np.squeeze(pictures[0].detach().numpy()), cmap=plt.cm.binary)
                plt.subplot(1, 2, 2)
                plt.imshow(np.squeeze(reconstruction.detach().numpy()[0]), cmap=plt.cm.binary)
                plt.show()

        # ----------------------------------------
        # Printing  progress
        # ----------------------------------------
        print('Epoch ', epoch, ' total cost: ', train_cost / counter)
        if epoch % 100 == 0:
            vae.eval()
            reconstruction, mu, std = vae(pictures)
            display_reconstruction(reconstruction.detach(), BATCH_SIZE)
            vae.train()
