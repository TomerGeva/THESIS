from ConfigDG import *
from database_functions import ModelManipulationFunctions
from classes.LoggerDG import LoggerDG
from ScatCoord_DG import import_data_sets_coord
from classes.TrainerDG import TrainerDG
import torch
from classes.DGcnn import ModDGCNN, ModDGCNN2
from auxiliary_functions import _init_


def main():
    mmf = ModelManipulationFunctions()
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logdir = _init_(PATH_LOGS)
    logger = LoggerDG(logdir=logdir)
    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ================================================================================
    # Importing the database
    # ================================================================================
    train_loader, test_loaders, thresholds = import_data_sets_coord(PATH_DATABASE_TRAIN,
                                                                    PATH_DATABASE_TEST,
                                                                    BATCH_SIZE,
                                                                    abs_sens=ABS_SENS,
                                                                    coord_mean=0,
                                                                    coord_scale=1,
                                                                    n_points=NUM_OF_POINTS,
                                                                    num_workers=NUM_WORKERS
                                                                    )
    # ================================================================================
    # Creating the model
    # ================================================================================
    # model = ModPointNet(device, POINTNET_TOPOLOGY)
    model = ModDGCNN(device, DGCNN_TOPOLOGY, CONCAT_EDGECONV, FLATTEN_TYPE)
    # model = ModDGCNN2(device, DGCNN_TOPOLOGY, FLATTEN_TYPE)
    # model = ModDGCNN2(device, MODGCNN_TOPOLOGY, FLATTEN_TYPE)
    mmf.initialize_weights(model, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD, method='xavier')
    model.to(device)
    # ================================================================================
    # Creating the trainer
    # ================================================================================
    norm_sens = (SENS_MEAN, SENS_STD) if NORM_SENS else (0, 1)
    trainer = TrainerDG(model,
                        lr=LR,
                        mom=MOM,
                        sched_step=SCHEDULER_STEP,
                        sched_gamma=SCHEDULER_GAMMA,
                        grad_clip=GRAD_CLIP,
                        group_thresholds=thresholds,
                        group_weights=MSE_GROUP_WEIGHT,
                        abs_sens=ABS_SENS,
                        norm_sens=norm_sens)
    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(model, train_loader, test_loaders, logger, epochs=EPOCH_NUM, save_per_epochs=40)


if __name__ == '__main__':
    main()






