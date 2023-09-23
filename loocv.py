import argparse
import os
from turtle import forward
import numpy as np
import pandas as pd
import scipy.io as scio

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from models.XRFEGAN import XRFEGAN
from utils.share import select_device
from datasets.LOOCVDataset import LOOCVDataset, extract_datas


def loocv_train(args: argparse.ArgumentParser) -> None:
    # device.
    device = select_device(args.cuda_use)

    # 
    outroot = args.outputs

    # extract datas.
    xrfDatas, xrfDatas_n, xrfCleanDatas, xrfCleanDatas_n, thetas_noisy, thetas_clean, min_value_noisy, min_value_clean, backgrounds = extract_datas(args.noisy_set, args.clean_set)

    all_results_Noisy = []
    all_results_Clean = []
    all_results_Genh = []
    all_results_Background = []
    all_results_Noisy_log = []
    all_results_Clean_log = []
    all_results_Genh_log = []

    # leave-one-out CV
    for i in tqdm(range(len(xrfDatas))):

        args.outputs = os.path.join(outroot, f"model_{i+1}")

        # delete.
        energys = xrfDatas.copy()
        energys_n = xrfDatas_n.copy()
        energys_clean = xrfCleanDatas.copy()
        energys_clean_n = xrfCleanDatas_n.copy()
        thetas_noisy_ = thetas_noisy.copy()
        thetas_clean_ = thetas_clean.copy()
        min_value_noisy_ = min_value_noisy.copy()
        min_value_clean_ = min_value_clean.copy()
        backgrounds_ = backgrounds.copy()

        energys = np.delete(energys, i, axis=0)
        energys_n = np.delete(energys_n, i, axis=0)
        energys_clean = np.delete(energys_clean, i, axis=0)
        energys_clean_n = np.delete(energys_clean_n, i, axis=0)
        thetas_noisy_ = np.delete(thetas_noisy_, i, axis=0)
        thetas_clean_ = np.delete(thetas_clean_, i, axis=0)
        min_value_noisy_ = np.delete(min_value_noisy_, i, axis=0)
        min_value_clean_ = np.delete(min_value_clean_, i, axis=0)
        backgrounds_ = np.delete(backgrounds_, i, axis=0)

        # load model.
        if args.model_name == "XRFEGAN":
            model = XRFEGAN(args=args)
        model.model_to_device(device)

        # load generator and discriminator pretrain weights.
        if args.G_pretrained_weight is not None and args.G_pretrained_weight != "":
            try:
                model.Generator_load_Weight(args.G_pretrained_weight, device)
            except Exception as e:
                raise "Generator load state dict eror"
        if args.D_pretrained_weight is not None and args.D_pretrained_weight != "":
            try:
                model.Discriminator_load_Weight(args.D_pretrained_weight, device)
            except Exception as e:
                raise "Discriminator load state dict eror"

        # train dataset
        train_dataset = LOOCVDataset(noisy=energys_n, 
                                     clean=energys_clean_n, 
                                     thetas_noisy=thetas_noisy_, 
                                     thetas_clean=thetas_clean_, 
                                     min_value_noisy=min_value_noisy_,
                                     min_value_clean=min_value_clean_,
                                     training=args.training)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

        # losser
        criterion = nn.MSELoss()

        # optimizer
        if args.optimizer == 'rmsprop':
            Goptimizer = optim.RMSprop(model.G.parameters(), lr=args.g_lr)
            Doptimizer = optim.RMSprop(model.D.parameters(), lr=args.d_lr)
        elif args.optimizer == 'adam':
            Goptimizer = optim.Adam(model.G.parameters(), lr=args.g_lr, betas=(0, 0.9))
            Doptimizer = optim.Adam(model.D.parameters(), lr=args.d_lr, betas=(0, 0.9))
        else:
            raise ValueError('Unrecognized optimizer {}'.format(args.optimizer))

        # tensorboard
        writer = SummaryWriter(os.path.join(args.outputs, 'train'))

        model.train(args=args, 
                    dataloader=train_dataloader,
                    criterion=criterion,
                    Goptimizer=Goptimizer,
                    Doptimizer=Doptimizer,
                    l1_weight=args.l1_weight,
                    l1_dec_step=args.l1_dec_step,
                    l1_dec_epoch=args.l1_dec_epoch,
                    writer=writer,
                    device=device)

        state_dict_path = os.path.join(args.outputs, 'weights', f'g_{args.epochs}.pt')

        writer.flush()
        writer.close()

        with torch.no_grad():
            # load generator weights.
            try:
                model.G.load_state_dict(torch.load(state_dict_path, map_location=device)['state_dict'])
            except Exception as e:
                raise "Generator load state dict eror"

            # test dataset
            test_dataset = LOOCVDataset(noisy=xrfDatas_n[i], 
                                        clean=xrfCleanDatas_n[i], 
                                        thetas_noisy=thetas_noisy[i], 
                                        thetas_clean=thetas_clean[i], 
                                        min_value_clean=min_value_noisy[i],
                                        min_value_noisy=min_value_clean[i],
                                        backgrounds=backgrounds[i],
                                        training=False)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

            os.makedirs(os.path.join(args.outputs, 'test'), exist_ok=True)

            results_Noisy, results_Clean, results_Genh, results_Background, results_Noisy_log, results_Clean_log, results_Genh_log \
                  = model.simulation_evaluate(dataloader=test_dataloader, save_dir=os.path.join(args.outputs, 'test'), device=device)
            
            pd.DataFrame(results_Noisy).to_csv(os.path.join(os.path.join(args.outputs, 'test'), 'noisy.csv'))
            pd.DataFrame(results_Clean).to_csv(os.path.join(os.path.join(args.outputs, 'test'), 'clean.csv'))
            pd.DataFrame(results_Genh).to_csv(os.path.join(os.path.join(args.outputs, 'test'), 'genh.csv'))
            pd.DataFrame(results_Background).to_csv(os.path.join(os.path.join(args.outputs, 'test'), 'background.csv'))
            pd.DataFrame(results_Noisy_log).to_csv(os.path.join(os.path.join(args.outputs, 'test'), 'noisy_log.csv'))
            pd.DataFrame(results_Clean_log).to_csv(os.path.join(os.path.join(args.outputs, 'test'), 'clean_log.csv'))
            pd.DataFrame(results_Genh_log).to_csv(os.path.join(os.path.join(args.outputs, 'test'), 'genh_log.csv'))

            results_Noisy = results_Noisy.tolist()
            results_Clean = results_Clean.tolist()
            results_Genh = results_Genh.tolist()
            results_Background = results_Background.tolist()
            results_Noisy_log = results_Noisy_log.tolist()
            results_Clean_log = results_Clean_log.tolist()
            results_Genh_log = results_Genh_log.tolist()


            scio.savemat(file_name=os.path.join(args.outputs, "test/simEvaluate.mat"), 
                        mdict={"results_Noisy": results_Noisy, 
                                'results_Clean': results_Clean, 
                                "results_Genh": results_Genh, 
                                "results_Background": results_Background,
                                "results_Noisy_log": results_Noisy_log,
                                "results_Clean_log": results_Clean_log,
                                "results_Genh_log": results_Genh_log})

            all_results_Noisy.append(results_Noisy[0])
            all_results_Clean.append(results_Clean[0])
            all_results_Genh.append(results_Genh[0])
            all_results_Background.append(results_Background[0])
            all_results_Noisy_log.append(results_Noisy_log[0])
            all_results_Clean_log.append(results_Clean_log[0])
            all_results_Genh_log.append(results_Genh_log[0])

    
    scio.savemat(file_name=os.path.join(outroot, "simEvaluate.mat"), 
                        mdict={"all_results_Noisy": all_results_Noisy, 
                                'all_results_Clean': all_results_Clean, 
                                "all_results_Genh": all_results_Genh, 
                                "all_results_Background": all_results_Background,
                                "all_results_Noisy_log": all_results_Noisy_log,
                                "all_results_Clean_log": all_results_Clean_log,
                                "all_results_Genh_log": all_results_Genh_log})
    all_results_Noisy = np.array(all_results_Noisy)
    all_results_Clean = np.array(all_results_Clean)
    all_results_Genh = np.array(all_results_Genh)
    all_results_Background = np.array(all_results_Background)
    all_results_Noisy_log = np.array(all_results_Noisy_log)
    all_results_Clean_log = np.array(all_results_Clean_log)
    all_results_Genh_log = np.array(all_results_Genh_log)
    pd.DataFrame(all_results_Noisy).to_csv(os.path.join(outroot, 'cv_noisy.csv'))
    pd.DataFrame(all_results_Clean).to_csv(os.path.join(outroot, 'cv_clean.csv'))
    pd.DataFrame(all_results_Genh).to_csv(os.path.join(outroot, 'cv_genh.csv'))
    pd.DataFrame(all_results_Background).to_csv(os.path.join(outroot, 'cv_background.csv'))
    pd.DataFrame(all_results_Noisy_log).to_csv(os.path.join(outroot, 'cv_noisy_log.csv'))
    pd.DataFrame(all_results_Clean_log).to_csv(os.path.join(outroot, 'cv_clean_log.csv'))
    pd.DataFrame(all_results_Genh_log).to_csv(os.path.join(outroot, 'cv_genh_log.csv'))
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_set', type=str, default="your pretraining noisy datas path (.mat)", help="noisy energy data directory")
    parser.add_argument('--clean_set', type=str, default="your pretraining clean datas path (.mat)", help="clean energy data directory")
    parser.add_argument('--model_name', type=str, default='XRFEGAN', help="energy enchance model name")
    parser.add_argument('--G_pretrained_weight', type=str, default='/`your project path`/outputs/pretrain/weights/g_1000.pt', help='Path to ckpt file to pre-load in training (Def: None).')
    parser.add_argument('--D_pretrained_weight', type=str, default=None, help='Path to ckpt file to pre-load in training (Def: None).')
    parser.add_argument('--epochs', type=int, default=2, help="train epoch")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--intervel', type=int, default=1, help="")
    parser.add_argument('--outputs', type=str, default="/`your project path`/outputs/loocv", help="outputs save directory")
    parser.add_argument('--cuda_use', type=bool, default=True, help="weather cuda (GPU) is used")
    parser.add_argument('--training', type=bool, default=True, help="training (True) or evaluate (False)")

    # Generator parameter
    parser.add_argument('--genc_fmaps', type=int, nargs='+',default=[64, 128, 256, 512, 1024], help='Number of G encoder feature maps.')
    parser.add_argument('--gkwidth', type=int, default=10, help="convolutional kernel, default (10*10)")
    parser.add_argument('--genc_poolings', type=int, nargs='+', default=[2, 2, 2, 2, 2], help='G encoder poolings')
    parser.add_argument('--z_dim', type=int, default=1024, help="gan noizy dim")
    parser.add_argument('--gdec_fmaps', type=int, nargs='+', default=None)
    parser.add_argument('--gdec_poolings', type=int, nargs='+', default=None, help='Optional dec poolings. Defaults to None so that encoder poolings are mirrored.')
    parser.add_argument('--gdec_kwidth', type=int, default=None, help="generator's dec convolutional kernel")
    parser.add_argument('--gnorm_type', type=str, default=None, help='Normalization to be used in G. Can be: (1) snorm, (2) bnorm or (3) none (Def: None).')
    parser.add_argument('--no_z', action='store_true', default=False)
    parser.add_argument('--no_skip', action='store_true', default=False)
    parser.add_argument('--pow_weight', type=float, default=0.001)
    parser.add_argument('--misalign_pair', action='store_true', default=False)
    parser.add_argument('--interf_pair', action='store_true', default=False)
    parser.add_argument('--skip_merge', type=str, default='concat', help="")
    parser.add_argument('--skip_type', type=str, default='alpha', help='Type of skip connection')
    parser.add_argument('--skip_init', type=str, default='one', help='Way to init skip connections (Def: one)')
    parser.add_argument('--skip_kwidth', type=int, default=11)
    parser.add_argument('--bias', action='store_true', default=True, help='Disable all biases in Generator')

    # Discriminator parameters
    parser.add_argument('--denc_fmaps', type=int, nargs='+', default=[64, 128, 256, 512, 1024], help='Number of D encoder feature maps, (Def: [64, 128, 256, 512, 1024]')
    parser.add_argument('--dpool_type', type=str, default='none', help='conv/none/gmax/gavg (Def: none)')
    parser.add_argument('--dpool_slen', type=int, default=16, help='Dimension of last conv D layer time axis prior to classifier real/fake (Def: 16)')
    parser.add_argument('--dkwidth', type=int, default=None, help='Disc kwidth (Def: None), None is gkwidth.')
    parser.add_argument('--denc_poolings', type=int, nargs='+', default=[2, 2, 2, 2, 2], help='(Def: [4, 4, 4, 4, 4])')
    parser.add_argument('--dnorm_type', type=str, default='bnorm', help='Normalization to be used in D. Can be: (1) snorm, (2) bnorm or (3) none (Def: bnorm).')
    parser.add_argument('--phase_shift', type=int, default=None)
    parser.add_argument('--sinc_conv', action='store_true', default=False)

    # other parameter.
    parser.add_argument('--optimizer', type=str, default='rmsprop')
    parser.add_argument('--g_lr', type=float, default=0.000005, help='Generator learning rate (Def: 0.00005).')
    parser.add_argument('--d_lr', type=float, default=0.00005, help='Discriminator learning rate (Def: 0.0005).')

    parser.add_argument('--l1_dec_epoch', type=int, default=100)
    parser.add_argument('--l1_weight', type=float, default=100, help='L1 regularization weight (Def. 100). ')
    parser.add_argument('--l1_dec_step', type=float, default=1e-5, help="")
    parser.add_argument('--reg_loss', type=str, default='l1_loss', help='Regression loss (l1_loss or mse_loss) in the output of G (Def: l1_loss)')
    parser.add_argument('--no_train_gen', action='store_true', default=False, help='Do NOT generate wav samples during training')
    parser.add_argument('--positions', type=list, default=[156, 664, 303, 304, 288, 232, 215, 249], help=' XRF peak positions')

    args = parser.parse_args()

    os.makedirs(args.outputs, exist_ok=True)

    loocv_train(args)
