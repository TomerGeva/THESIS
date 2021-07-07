import os
from torch.autograd import Variable
from ScatterCoordinateDataset import import_data_sets
from functions import load_state_train
from ConfigVAE import PATH_LOGS, BATCH_SIZE


def load_and_batch():
    """
    :return: This function loads a saves model, and tests the MSE of the target error
    """
    data_path                   = os.path.join(PATH_LOGS, '1_7_2021_15_50', 'VAE_model_data_lr_0.0003_epoch_299.tar')
    train_loader, test_loader   = import_data_sets(BATCH_SIZE, 0.15)
    mod_vae, trainer            = load_state_train(data_path)

    smapled_batch   = next(iter(test_loader))
    grids           = Variable(smapled_batch['grid'].float()).to(mod_vae.device)
    sensitivities   = Variable(smapled_batch['sensitivity'].float()).to(mod_vae.device)

    mod_vae.eval()
    outputs, mu, logvar = mod_vae(grids)
    print('Outputs: ' + str(outputs))
    print('Targets: ' + str(sensitivities))


if __name__ == '__main__':
    load_and_batch()
