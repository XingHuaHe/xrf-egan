import argparse
import os

import pandas as pd
import scipy.io as scio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from models.XRFEGAN import XRFEGAN
from utils.share import select_device
from datasets.XRFEDataset import XRFEDataset_dat, XRFEDataset_mat

def evaluate(args: argparse.ArgumentParser) -> None:
    """
        evaluate a xrf energy GAN enhance model.
    """

    # device
    device = select_device(args.cuda_use)

    # load model.
    if args.model_name == "XRFEGAN":
        model = XRFEGAN(args=args)
    model.model_to_device(device)

    # evaluate
    with torch.no_grad():
        # load generator weights.
        try:
            model.G.load_state_dict(torch.load(args.G_pretrained_weight, map_location=device)['state_dict'])
        except Exception as e:
            raise "Generator load state dict eror"

        # test dataset
        if os.path.basename(args.noisy_set).endswith('.mat'):
            # .mat data files.
            test_dataset = XRFEDataset_mat(args.noisy_set, args.clean_set, training=args.training)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        else:
            print("The inputed noisy and clean dataset must be mat format file.")

        os.makedirs(os.path.join(args.outputs, 'test'), exist_ok=True)

        results_Noisy, results_Clean, results_Genh, results_Background, results_Noisy_log, results_Clean_log, results_Genh_log \
            = model.evaluate(dataloader=test_dataloader, save_dir=os.path.join(args.outputs, 'test'), device=device)
        # pd.DataFrame(results).to_csv(os.path.join(os.path.join(args.outputs, 'test'), 'enhance_result_ss.csv'))

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_set', type=str, default="your pretraining noisy datas path (.mat)", help="noisy energy data directory")
    parser.add_argument('--clean_set', type=str, default="your pretraining clean datas path (.mat)", help="clean energy data directory")
    parser.add_argument('--model_name', type=str, default='XRFEGAN', help="energy enchance model name")
    parser.add_argument('--G_pretrained_weight', type=str, default="/`your project path`/outputs/fine-tuning/weights/g_1000.pt", help='Path to ckpt file to pre-load in training (Def: None).')
    parser.add_argument('--D_pretrained_weight', type=str, default=None, help='Path to ckpt file to pre-load in training (Def: None).')
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--outputs', type=str, default="./outputs/evaluate", help="outputs save directory")
    parser.add_argument('--cuda_use', type=bool, default=True, help="weather cuda (GPU) is used")
    parser.add_argument('--training', type=bool, default=False, help="training (True) or evaluate (False)")

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
    parser.add_argument('--g_lr', type=float, default=0.00005, help='Generator learning rate (Def: 0.00005).')
    parser.add_argument('--d_lr', type=float, default=0.00005, help='Discriminator learning rate (Def: 0.0005).')

    parser.add_argument('--l1_dec_epoch', type=int, default=100)
    parser.add_argument('--l1_weight', type=float, default=100, help='L1 regularization weight (Def. 100). ')
    parser.add_argument('--l1_dec_step', type=float, default=1e-5, help="")
    parser.add_argument('--reg_loss', type=str, default='l1_loss', help='Regression loss (l1_loss or mse_loss) in the output of G (Def: l1_loss)')
    parser.add_argument('--no_train_gen', action='store_true', default=False, help='Do NOT generate wav samples during training')
    parser.add_argument('--positions', type=list, default=[156, 664, 303, 304, 288, 232, 215, 249], help=' XRF peak positions')

    args = parser.parse_args()

    os.makedirs(args.outputs, exist_ok=True)

    evaluate(args)