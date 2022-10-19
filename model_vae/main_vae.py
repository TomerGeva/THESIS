# import numpy as np
# import os
import torch
import os
import matplotlib.pyplot as plt
from ConfigVAE                  import *
from classes.LoggerVAE import LoggerVAE
from classes.LoggerLatent import LoggerLatent
from classes.TrainerVAE import TrainerVAE
from classes.TrainerLatent import TrainerLatent
from classes.ModVAE import ModVAE
from auxiliary_functions        import PlottingFunctions, _init_
from ScatterCoordinateDataset   import import_data_sets_pics
from ScatCoord_DG               import import_data_sets_coord
from global_const               import encoder_type_e
from database_functions         import ModelManipulationFunctions, PathFindingFunctions
from blob_detection_functions   import BlobDetectionFunctions
from database_functions         import DatabaseFunctions


def main_vae(encoder_type=encoder_type_e.DENSE,
             load_model=None,
             start_epoch=0,
             copy_weights=None,
             copy_weights_epoch=0):
    pf  = PlottingFunctions()
    pff = PathFindingFunctions()
    mmf = ModelManipulationFunctions()
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logdir = _init_(PATH_LOGS)
    logger = LoggerVAE(logdir=logdir)

    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ================================================================================
    # Importing the data
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
    if load_model is None:
        # ============================================================================
        # Creating the net & trainer objects
        # ============================================================================
        if encoder_type == encoder_type_e.DENSE:
            encoder_topology = DENSE_ENCODER_TOPOLOGY
            decoder_topology = DECODER_TOPOLOGY
        elif encoder_type == encoder_type_e.VGG:
            encoder_topology = VGG_ENCODER_TOPOLOGY
            decoder_topology = DECODER_TOPOLOGY
        elif encoder_type == encoder_type_e.SEPARABLE:
            encoder_topology = SEPARABLE_ENCODER_TOPOLOGY
            decoder_topology = DECODER_TOPOLOGY
        elif encoder_type == encoder_type_e.TANSFORMER:
            encoder_topology = TRANS_ENCODER_TOPOLOGY
            decoder_topology = DECODER_TOPOLOGY
        elif encoder_type == encoder_type_e.RES_VGG:
            encoder_topology = VGG_RES_ENCODER_TOPOLOGY
            decoder_topology = RES_DECODER_TOPOLOGY
        elif encoder_type == encoder_type_e.FULLY_CONNECTED:
            encoder_topology = FC_ENCODER_TOPOLOGY
            decoder_topology = FC_DECODER_TOPOLOGY
        else:
            raise ValueError('Unknown encoder type inputted')

        mod_vae = ModVAE(device=device,
                         encoder_topology=encoder_topology,
                         decoder_topology=decoder_topology,
                         latent_space_dim=LATENT_SPACE_DIM,
                         encoder_type=encoder_type,
                         mode=MODE,
                         model_out=MODEL_OUT)
        mmf.initialize_weights(mod_vae, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD, method='xavier')
        mod_vae.to(device)  # allocating the computation to the CPU or GPU
        # ================================================================================
        # Copy weights if needed
        # ================================================================================
        if copy_weights is not None:
            # ------------------------------------------------------------------------
            # Loading the wanted model
            # ------------------------------------------------------------------------
            chosen_file = pff.get_full_path(copy_weights, copy_weights_epoch)
            mod_vae_source, _ = mmf.load_state_train(chosen_file)
            # ------------------------------------------------------------------------
            # Copying weights
            # ------------------------------------------------------------------------
            mmf.copy_net_weights(mod_vae_source, mod_vae)
            print('Copied weights, lets see what happens . . .')
        # ================================================================================
        # Creating trainer
        # ================================================================================
        if encoder_type == encoder_type_e.FULLY_CONNECTED:
            trainer = TrainerVAE(mod_vae,
                                 lr=LR,
                                 mom=MOM,
                                 beta_dkl=BETA_DKL,
                                 beta_grid=BETA_GRID,
                                 sched_step=SCHEDULER_STEP,
                                 sched_gamma=SCHEDULER_GAMMA,
                                 grad_clip=GRAD_CLIP,
                                 group_thresholds=thresholds,  # sens cost
                                 group_weights=MSE_GROUP_WEIGHT,  # sens cost
                                 abs_sens=ABS_SENS, norm_sens=norm_sens,
                                 xquantize=XQUANTIZE, yquantize=YQUANTIZE, n=N, coord2map_sigma=COORD2MAP_SIGMA)
        else:
            trainer = TrainerVAE(mod_vae,
                                 lr=LR,
                                 mom=MOM,
                                 beta_dkl=BETA_DKL,
                                 beta_grid=BETA_GRID,
                                 sched_step=SCHEDULER_STEP,
                                 sched_gamma=SCHEDULER_GAMMA,
                                 grad_clip=GRAD_CLIP,
                                 group_thresholds=thresholds,  # sens cost
                                 group_weights=MSE_GROUP_WEIGHT,  # sens cost
                                 abs_sens=ABS_SENS, norm_sens=norm_sens,
                                 grid_pos_weight=GRID_POS_WEIGHT,
                                 xquantize=XQUANTIZE, yquantize=YQUANTIZE)
    else:
        print('Loading model . . .')
        # ==============================================================================
        # Extracting the full file path
        # ==============================================================================
        chosen_file = pff.get_full_path(load_model, start_epoch)
        # ==============================================================================
        # Loading the needed models and data
        # ==============================================================================
        mod_vae, trainer = mmf.load_state_train(chosen_file, thresholds=thresholds)
    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(mod_vae, train_loader, test_loaders, logger, save_per_epochs=20)


def main_vae_pcloud(encoder_type=encoder_type_e.DENSE, load_model=None, start_epoch=0):
    pf = PlottingFunctions()
    pff = PathFindingFunctions()
    mmf = ModelManipulationFunctions()
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logdir = _init_(PATH_LOGS)
    logger = LoggerVAE(logdir=logdir)
    # ================================================================================
    # Allocating device of computation: CPU or GPU
    # ================================================================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ================================================================================
    # Importing the data
    # ================================================================================
    norm_coord = (COORD_MEAN, COORD_SCALE) if NORM_GRID else (0, 1)
    norm_sens  = (SENS_MEAN, SENS_STD) if NORM_SENS else (0, 1)
    train_loader, test_loaders, thresholds = import_data_sets_coord(PATH_DATABASE_TRAIN,
                                                                    PATH_DATABASE_TEST,
                                                                    BATCH_SIZE,
                                                                    abs_sens=ABS_SENS,
                                                                    coord_mean=norm_coord[0],
                                                                    coord_scale=norm_coord[1],
                                                                    num_workers=NUM_WORKERS
                                                                    )
    if load_model is None:
        # ============================================================================
        # Creating the net & trainer objects
        # ============================================================================
        if encoder_type == encoder_type_e.PCLOUD_GRAPH:
            encoder_topology = MODGCNN_ENCODER_TOPOLOGY
            decoder_topology = FC_DECODER_TOPOLOGY
        else:
            raise ValueError('Unknown encoder type inputted')
        model = ModVAE(device=device,
                       encoder_topology=encoder_topology,
                       decoder_topology=decoder_topology,
                       latent_space_dim=LATENT_SPACE_DIM,
                       encoder_type=encoder_type,
                       mode=MODE,
                       model_out=MODEL_OUT, flatten_type=FLATTEN_TYPE)
        mmf.initialize_weights(model, INIT_WEIGHT_MEAN, INIT_WEIGHT_STD, method='xavier')
        model.to(device)  # allocating the computation to the CPU or GPU
        trainer = TrainerVAE(model, lr=LR, mom=MOM, beta_dkl=BETA_DKL, beta_grid=BETA_GRID,
                             sched_step=SCHEDULER_STEP, sched_gamma=SCHEDULER_GAMMA,
                             grad_clip=GRAD_CLIP,
                             group_thresholds=thresholds,  # sens cost
                             group_weights=MSE_GROUP_WEIGHT,  # sens cost
                             abs_sens=ABS_SENS, norm_sens=norm_sens,
                             xquantize=XQUANTIZE, yquantize=YQUANTIZE, n=N, coord2map_sigma=COORD2MAP_SIGMA)
    # ================================================================================
    # Training
    # ================================================================================
    trainer.train(model, train_loader, test_loaders, logger, save_per_epochs=40)


def main_optim_input(path=None, epoch=None):
    pf  = PlottingFunctions()
    pff = PathFindingFunctions()
    mmf = ModelManipulationFunctions()
    # ================================================================================
    # creating full file path
    # ================================================================================
    chosen_file = pff.get_full_path(path, epoch=epoch)
    # ================================================================================
    # Loading the decoder creating the input vector
    # ================================================================================
    mod_vae, _ = mmf.load_state_train(chosen_file)
    mod_vae.mode = mode_e.AUTOENCODER
    mod_vae.model_out = model_output_e.SENS
    mod_vae.decoder.model_out = model_output_e.SENS
    mod_vae.requires_grad_(False)

    input_mat  = torch.nn.Parameter(torch.randn([1, 1, XQUANTIZE, YQUANTIZE], device=mod_vae.device), requires_grad=True)
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logger = LoggerLatent(logdir=path, filename='logger_latent.txt')
    # ================================================================================
    # Creating the trainer object
    # ================================================================================
    trainer = TrainerLatent(input_vec=input_mat,
                            lr=LR,
                            mom=MOM,
                            beta=BETA_DKL,
                            sched_step=SCHEDULER_STEP,
                            sched_gamma=SCHEDULER_GAMMA,
                            grad_clip=GRAD_CLIP,
                            abs_sens=ABS_SENS,
                            sens_std=SENS_STD,
                            sens_mean=SENS_MEAN)
    # ================================================================================
    # Training
    # ================================================================================
    optim_mat = trainer.optimize_input(input_mat, mod_vae, 2000, logger, save_per_epoch=1)
    pf.plot_grid(optim_mat)


def main_optim_latent(path=None, epoch=None):
    pf = PlottingFunctions()
    pff = PathFindingFunctions()
    mmf = ModelManipulationFunctions()
    # ================================================================================
    # creating full file path
    # ================================================================================
    chosen_file = pff.get_full_path(path, epoch=epoch)
    # ================================================================================
    # Loading the decoder creating the input vector
    # ================================================================================
    decoder, latent_dim = mmf.load_decoder(data_path=chosen_file)
    input_vec  = torch.nn.Parameter(torch.randn([1, latent_dim], device=decoder.device), requires_grad=True)
    # ================================================================================
    # Setting the logger
    # ================================================================================
    logger = LoggerLatent(logdir=path)
    # ================================================================================
    # Creating the trainer object
    # ================================================================================
    trainer = TrainerLatent(input_vec=input_vec,
                            lr=LR,
                            mom=MOM,
                            beta=BETA_DKL,
                            sched_step=SCHEDULER_STEP,
                            sched_gamma=SCHEDULER_GAMMA,
                            grad_clip=GRAD_CLIP,
                            abs_sens=ABS_SENS,
                            sens_std=SENS_STD,
                            sens_mean=SENS_MEAN)
    # ================================================================================
    # Training
    # ================================================================================
    opt_vec = trainer.optimize_input(input_vec, decoder, 50000, logger, save_per_epoch=1)
    # ================================================================================
    # Plotting optimized latent space vector
    # ================================================================================
    plt.plot(opt_vec.cpu().detach().numpy().squeeze())
    plt.title('Optimized Latent Vector')
    plt.xlabel('index')
    plt.grid()
    plt.show()

    pass_through_save_scat(opt_vec, decoder, path, sigma_0=0.3, scale=1.15, k=15, peak_threshold=3, kernel_size=25)


def pass_through_save_scat(latent_vec, decoder, path, sigma_0=0.3, scale=1.15, k=15, peak_threshold=3, kernel_size=25):
    # ==============================================================================================================
    # Local variables
    # ==============================================================================================================
    sigmoid = torch.nn.Sigmoid()
    pf = PlottingFunctions()
    dbf = DatabaseFunctions()
    pff = PathFindingFunctions()
    mmf = ModelManipulationFunctions()
    bdf = BlobDetectionFunctions(peak_threshold=peak_threshold,
                                 kernel_size=kernel_size,
                                 sigma_0=sigma_0,
                                 scale=scale,
                                 k=k)
    x_rate = (XRANGE[1] - XRANGE[0] + 1) / XQUANTIZE
    y_rate = (YRANGE[1] - YRANGE[0] + 1) / YQUANTIZE
    dmin = DMIN
    threshold = 0.35
    # ==============================================================================================================
    # No grad for speed
    # ==============================================================================================================
    with torch.no_grad():
        # ------------------------------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------------------------------
        grid_out, sens_out = decoder(latent_vec)
        grid_out = np.squeeze(sigmoid(grid_out).cpu().detach().numpy())
        grid_out_sliced = mmf.slice_grid(grid_out, threshold)
    # ==============================================================================================================
    # Creating scale space and DoG space
    # ==============================================================================================================
    print('Computing scale space . . . ')
    scale_space = bdf.create_scale_space(grid_out)
    scale_space_sliced = bdf.create_scale_space(grid_out_sliced)
    print('Computing Difference of Gaussians space . . . ')
    dog_space = bdf.create_dog_space(scale_space)
    dog_space_sliced = bdf.create_dog_space(scale_space_sliced)
    print('Finding local maxima . .. ')
    local_max = bdf.extract_local_maxima(dog_space)
    local_max_sliced = bdf.extract_local_maxima(dog_space_sliced)
    # ==============================================================================================================
    # Removing the cylinders based on the minimal distance and blob size
    # ==============================================================================================================
    print('Making array valid . . .')
    valid_array = dbf.check_array_validity(local_max, x_rate=x_rate, y_rate=y_rate, dmin=dmin)
    valid_array_sliced = dbf.check_array_validity(local_max_sliced, x_rate=x_rate, y_rate=y_rate, dmin=dmin)
    print('Valid array saved to ' + os.path.join(path, PP_DATA))
    dbf.save_array(valid_array, (sens_out.item() * SENS_STD) + SENS_MEAN, os.path.join(path, PP_DATA),
                   name='scatter_raw.csv')
    dbf.save_array(valid_array_sliced, (sens_out.item() * SENS_STD) + SENS_MEAN, os.path.join(path, PP_DATA),
                   name='scatter_sliced.csv')
    # ==============================================================================================================
    # Plotting
    # ==============================================================================================================
    plt.figure()
    plt.imshow(1 - grid_out, cmap='gray')
    plt.scatter(valid_array[:, 0], valid_array[:, 1])
    plt.legend(['original', 'reconstructed', 'original unique', 'reconstructed unique'])
    plt.title('Raw Output')

    plt.figure()
    plt.imshow(1 - grid_out_sliced, cmap='gray')
    plt.scatter(valid_array_sliced[:, 0], valid_array_sliced[:, 1])
    plt.legend(['original', 'reconstructed', 'original unique', 'reconstructed unique'])
    plt.title('Slicer at ' + str(threshold))
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(SEED)
    phase = 1
    # ================================================================================
    # Training VAE on scatterer arrays and matching sensitivities
    # ================================================================================
    if phase == 1:

        load_path   = None
        copy_path   = None
        load_epoch  = 1000
        copy_epoch  = 320
        # load_path   = '..\\results_vae\\5_8_2022_8_41'
        # load_path   = '..\\results\\9_4_2022_18_43'
        # load_path   = '..\\results_vae\\8_9_2022_14_38'
        # load_path   = '..\\results_vae\\20_9_2022_10_39'
        # load_path   = '..\\results\\9_6_2022_8_28'
        # copy_path   = '..\\results\\15_12_2021_23_46'
        if ENCODER_TYPE == encoder_type_e.PCLOUD_GRAPH:
            main_vae_pcloud(ENCODER_TYPE, load_path)
        else:
            main_vae(ENCODER_TYPE,
                     load_model=load_path, start_epoch=load_epoch,
                     copy_weights=copy_path, copy_weights_epoch=copy_epoch)
    # ================================================================================
    # Using the decoder to maximize sensitivity prediction
    # ================================================================================
    if phase == 2:
        c_path = '..\\results\\16_1_2022_21_39'
        c_epoch = 700
        # main_optim_input(path=c_path, epoch=epoch)
        main_optim_latent(path=c_path, epoch=c_epoch)

