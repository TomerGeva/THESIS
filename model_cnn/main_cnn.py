from ConfigCNN import *
import torch
from database_functions import ModelManipulationFunctions
from ScatterCoordinateDataset import import_data_sets_pics
from classes.LoggerCNN import LoggerCNN
from classes.TrainerCnn import TrainerCNN
from classes.Model_CNN import CnnModel
from auxiliary_functions import _init_
from model_cnn.utils import get_topology


def main():
    mmf = ModelManipulationFunctions()
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logdir = _init_(PATH_LOGS)
    logger = LoggerCNN(logdir=logdir)
    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ================================================================================
    # Importing the database
    # ================================================================================
    norm_grid = (GRID_MEAN, GRID_STD) if NORM_GRID else (0, 1)
    norm_sens = (SENS_MEAN, SENS_STD) if NORM_SENS else (0, 1)
    train_loader, test_loaders, thresholds = import_data_sets_pics(PATH_DATABASE_TRAIN,
                                                                   PATH_DATABASE_TEST,
                                                                   BATCH_SIZE,
                                                                   abs_sens=ABS_SENS,
                                                                   dilation=DILATION,
                                                                   norm_sens=norm_sens,
                                                                   norm_grid=norm_grid,
                                                                   num_workers=NUM_WORKERS)
    # ================================================================================
    # Creating the model
    # ================================================================================
    model = CnnModel(device, get_topology(), MODEL_TYPE)
    mmf.initialize_weights(model, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD, method='xavier')
    model.to(device)
    # ================================================================================
    # Creating the trainer
    # ================================================================================
    trainer = TrainerCNN(model,
                         lr=LR,
                         mom=MOM,
                         sched_step=SCHEDULER_STEP,
                         sched_gamma=SCHEDULER_GAMMA,
                         grad_clip=GRAD_CLIP,
                         group_thresholds=thresholds,
                         group_weights=MSE_GROUP_WEIGHT,
                         abs_sens=ABS_SENS,
                         norm_sens=norm_sens,
                         xquantize=XQUANTIZE,
                         yquantize=YQUANTIZE)
    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(model, train_loader, test_loaders, logger, save_per_epochs=SAVE_PER_EPOCH)


if __name__ == '__main__':
    main()






