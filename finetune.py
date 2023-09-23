
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from models.XRFEGAN import XRFEGAN
from utils.share import select_device
from datasets.XRFEDataset import XRFEDataset_dat, XRFEDataset_mat

def finetune(args: argparse.ArgumentParser) -> None:
    """
        training a xrf energy GAN enhance model.
    """

    # device
    device = select_device(args.cuda_use)

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

    # finetune dataset
    if os.path.basename(args.noisy_set).endswith('.mat'):
        train_dataset = XRFEDataset_mat(args.noisy_set, args.clean_set, training=args.training)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        print("The inputed noisy and clean dataset must be mat format file.")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_set', type=str, default="your pretraining noisy datas path (.mat)", help="noisy energy data directory")
    parser.add_argument('--clean_set', type=str, default="your pretraining noisy datas path (.mat)", help="clean energy data directory")
    parser.add_argument('--model_name', type=str, default='XRFEGAN', help="energy enchance model name")
    parser.add_argument('--G_pretrained_weight', type=str, default='/`your project path`/outputs/pretrain/weights/g_1000.pt', help='Path to ckpt file to pre-load in training (Def: None).')
    parser.add_argument('--D_pretrained_weight', type=str, default=None, help='Path to ckpt file to pre-load in training (Def: None).')
    parser.add_argument('--epochs', type=int, default=500, help="train epoch")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--intervel', type=int, default=1, help="")
    parser.add_argument('--outputs', type=str, default="/`your project path`/outputs/fine-tuning", help="outputs save directory")
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

    finetune(args)
