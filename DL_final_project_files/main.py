from database_functions import *
from convolution_net.Config import *


def main():
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logger = Logger(PATH_LOGS)

    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ================================================================================
    # Fetching the train and test data
    # ================================================================================
    train_loader, test_loader = import_data_loaders(BATCH_SIZE)

    # ================================================================================
    # Training a new net
    # ================================================================================
    if TRAIN_NEW_NET:
        train_new_net(logger, train_loader, test_loader, device)

    # ================================================================================
    # Loading a trained model and testing the accuracy
    # ================================================================================
    if LOAD_AND_TEST:
        # ----------------------------------------------------------------------------
        # Loading the network and trainer
        # ----------------------------------------------------------------------------
        net, trainer = load_state_train()
        net.eval()

        # ================================================================================
        # Testing accuracy
        # ================================================================================
        train_accuracy = accuracy_test(0, net, train_loader)
        test_accuracy = accuracy_test(0, net, test_loader)
        # ================================================================================
        # Printing the results
        # ================================================================================
        print("Train accuracy: {:.6f}% \tTest accuracy: {:.6f}%".format(train_accuracy, test_accuracy))

    logger.logger.close()


if __name__ == '__main__':
    main()
