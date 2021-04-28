from Config import *
import torch
from functions_toy import import_data_sets, train, initialize_weights_toy
from VAE_toy import VaeToy


def main():
    LABELS_DICT = {0: 'T-Shirt',
                   1: 'Trouser',
                   2: 'Pullover',
                   3: 'Dress',
                   4: 'Coat',
                   5: 'Sandal',
                   6: 'Shirt',
                   7: 'Sneaker',
                   8: 'Bag',
                   9: 'Ankle Boot'}  # label dictionary

    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ================================================================================
    # Fetching the train and test data
    # ================================================================================
    train_loader, test_loader = import_data_sets(BATCH_SIZE)

    # ================================================================================
    # Creating the VAE object
    # ================================================================================
    vae = VaeToy().to(device)
    initialize_weights_toy(vae, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD)
    optimizer = torch.optim.Adam(vae.parameters(), lr=3e-4, betas=(0.5, 0.999))

    # ================================================================================
    # Training the VAE
    # ================================================================================
    train(vae, optimizer, train_loader, test_loader, device)

    print('hi')


if __name__ == '__main__':
    main()
