from argparse import ArgumentParser
import os
from typing import Union
from typing import Type
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.generator import Generator
from models.discriminator import Discriminator
from utils.utils import weights_init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class XRFEGAN():
    """
        XRF enchance GAN model.
    """

    def __init__(self, args: ArgumentParser, generator=None, discriminator=None) -> None:
        super().__init__()

        self.outputs = args.outputs
        self.positions = args.positions

        if generator is None:
            self.G = Generator(ninputs=1,
                               fmaps=args.genc_fmaps,
                               kwidth=args.gkwidth,
                               poolings=args.genc_poolings,
                               z_dim=args.z_dim,
                               dec_fmaps=args.gdec_fmaps,
                               dec_kwidth=args.gdec_kwidth,
                               dec_poolings=args.gdec_poolings,

                               no_z=args.no_z,
                               skip=(not args.no_skip),
                               bias=args.bias,
                               skip_init=args.skip_init,
                               skip_type=args.skip_type,
                               skip_merge=args.skip_merge,
                               skip_kwidth=args.skip_kwidth)
        else:
            self.G = generator
        self.G.apply(weights_init)

        if discriminator is None:
            dkwidth = args.gkwidth if args.dkwidth is None else args.dkwidth
            self.D = Discriminator(ninputs=2, 
                                   fmaps=args.denc_fmaps,
                                   kwidth=dkwidth,
                                   poolings=args.denc_poolings,
                                   pool_type=args.dpool_type,
                                   pool_slen=args.dpool_slen, 
                                   norm_type=args.dnorm_type,
                                   phase_shift=args.phase_shift)
        else:
            self.D = discriminator
        self.D.apply(weights_init)

    
    def Generator_load_Weight(self, weight: str, device: torch.device) -> None:
        self.G.load_state_dict(torch.load(weight, map_location=device)['state_dict'])

    def Discriminator_load_Weight(self, weight: str, device: torch.device) -> None:
        self.D.load_state_dict(torch.load(weight, map_location=device))

    def model_to_device(self, device) -> None:
        self.G.to(device=device)
        self.D.to(device=device)

    def infer_G(self, nwav, cwav=None, z=None, ret_hid=False):
        if ret_hid:
            Genh, hall = self.G(nwav, z=z, ret_hid=ret_hid)
            return Genh, hall
        else:
            Genh = self.G(nwav, z=z, ret_hid=ret_hid)
            return Genh

    def infer_D(self, x_, ref):
        D_in = torch.cat((x_, ref), dim=1)
        return self.D(D_in)

    def simulation_train(self, 
              args: ArgumentParser, 
              dataloader: DataLoader, 
              criterion, 
              Goptimizer,
              Doptimizer,
              l1_weight, 
              l1_dec_step,
              l1_dec_epoch,
              writer: SummaryWriter = None,
              device=torch.device) -> None:


        self.reg_loss = getattr(F, args.reg_loss) # l1_loss
        self.mse_loss = getattr(F, "mse_loss")
        iteration = 1
        z_sample = None

        label = torch.ones(args.batch_size)
        label = label.to(device)
        for epoch in tqdm(range(1, args.epochs + 1)):
            self.G.train()
            self.D.train()
            for ind, (noisy, clean, _, _, _, _) in enumerate(dataloader, start=1):
                if epoch >= l1_dec_epoch:
                    if l1_weight > 0:
                        l1_weight -= l1_dec_step
                        l1_weight = max(0, l1_weight) # ensure it is 0 if it goes < 0

                noisy = noisy.type(torch.FloatTensor).unsqueeze(1)
                clean = clean.type(torch.FloatTensor).unsqueeze(1)
                label.resize_(clean.size(0)).fill_(1)
                clean = clean.to(device)
                noisy = noisy.to(device)

                # (1) D real update
                Doptimizer.zero_grad()
                total_d_fake_loss = 0
                total_d_real_loss = 0
                Genh = self.infer_G(noisy, clean)
                lab = label
                d_real, _ = self.infer_D(clean, noisy)
                d_real_loss = criterion(d_real.view(-1), lab)
                d_real_loss.backward()
                total_d_real_loss += d_real_loss

                # (2) D fake update
                d_fake, _ = self.infer_D(Genh.detach(), noisy)
                lab = label.fill_(0)
                d_fake_loss = criterion(d_fake.view(-1), lab)
                d_fake_loss.backward()
                total_d_fake_loss += d_fake_loss
                Doptimizer.step()

                d_loss = d_fake_loss + d_real_loss

                # (3) G real update
                Goptimizer.zero_grad()
                lab = label.fill_(1)
                d_fake_, _ = self.infer_D(Genh, noisy)
                g_adv_loss = criterion(d_fake_.view(-1), lab)
                #g_l1_loss = l1_weight * F.l1_loss(Genh, clean)
                g_l1_loss = l1_weight * self.reg_loss(Genh, clean)

                # local mse loss
                g_mse_loss = self.mse_loss(Genh, clean)
                for i in range(len(self.positions)):
                    g_mse_loss += self.mse_loss(Genh[:, :, self.positions[i]-30:self.positions[i]+30], clean[:, :, self.positions[i]-30:self.positions[i]+30])

                g_loss = g_adv_loss + g_l1_loss + 100 * g_mse_loss
                g_loss.backward()
                Goptimizer.step()

                if z_sample is None and not self.G.no_z:
                    # capture sample now that we know shape after first inference
                    z_sample = self.G.z[:20, :, :].contiguous()
                    print('z_sample size: ', z_sample.size())
                    z_sample = z_sample.to(device)
                if ind % 3 == 0 or ind >= len(dataloader):
                    d_real_loss_v = d_real_loss.cpu().item()
                    d_fake_loss_v = d_fake_loss.cpu().item()
                    g_adv_loss_v = g_adv_loss.cpu().item()
                    g_l1_loss_v = g_l1_loss.cpu().item()
                    g_mse_loss_v = g_mse_loss.cpu().item()

                    log = '(Iter {}) Batch {}/{} (Epoch {}) d_real:{:.4f}, d_fake:{:.4f}, '.format(iteration, ind, len(dataloader), epoch, d_real_loss_v, d_fake_loss_v)
                    log += 'g_adv:{:.4f}, g_l1:{:.4f}, g_local:{:.4f}, l1_w: {:.2f}, btime: {:.4f} s, mbtime: {:.4f} s'.format(g_adv_loss_v, g_l1_loss_v, g_mse_loss_v, l1_weight, 0, 0)
                    # print(log)

                    writer.add_scalar('D_real', d_real_loss_v, iteration)
                    writer.add_scalar('D_fake', d_fake_loss_v, iteration)
                    writer.add_scalar('G_adv', g_adv_loss_v, iteration)
                    writer.add_scalar('G_l1', g_l1_loss_v, iteration)
                    writer.add_scalar('G_local', g_mse_loss_v, iteration)
                    writer.add_histogram('D_fake__hist', d_fake_.cpu().data, iteration, bins='sturges')
                    writer.add_histogram('D_fake_hist', d_fake.cpu().data, iteration, bins='sturges')
                    writer.add_histogram('D_real_hist', d_real.cpu().data, iteration, bins='sturges')
                    writer.add_histogram('Gz', Genh.cpu().data, iteration, bins='sturges')
                    writer.add_histogram('clean', clean.cpu().data, iteration, bins='sturges')
                    writer.add_histogram('noisy', noisy.cpu().data, iteration, bins='sturges')
                    # get D and G weights and plot their norms by layer and global
                    def model_weights_norm(model, total_name):
                        total_GW_norm = 0
                        for k, v in model.named_parameters():
                            if 'weight' in k:
                                W = v.data
                                W_norm = torch.norm(W)
                                writer.add_scalar('{}_Wnorm'.format(k), W_norm, iteration)
                                total_GW_norm += W_norm
                        writer.add_scalar('{}_Wnorm'.format(total_name), total_GW_norm, iteration)
                    model_weights_norm(self.G, 'Gtotal')
                    model_weights_norm(self.D, 'Dtotal')
                iteration += 1
            
            # save models in end of epoch with EOE savers
            if epoch % 10 == 0 or epoch >= args.epochs:
                self.save(save_path=os.path.join(self.outputs, 'weights'),
                          step=epoch,
                          Goptimizer=Goptimizer, 
                          Doptimizer=Doptimizer)

    def simulation_evaluate(self, dataloader, save_dir, do_noisy=False, max_samples=1, device='cpu') -> np.array:
        r"""
            evaluate model and show the effect. 
        """

        self.G.eval()
        self.D.eval()

        results_Noisy_log = []
        results_Genh_log = []
        results_Clean_log = []

        results_Noisy = []
        results_Genh = []
        results_Clean = []
        results_Background = []
        
        # going over dataset ONCE
        for ind, (noisy, clean, thetas_noisy, thetas_clean, min_value_noisy, min_value_clean, background) in tqdm(enumerate(dataloader, start=1)):
            noisy = noisy.type(torch.FloatTensor).unsqueeze(1)

            clean = clean.to(device)
            noisy = noisy.to(device)
            background = background.to(device)
            Genh = self.infer_G(noisy).squeeze(1)

            noisy_log_npy = noisy.squeeze(1).cpu().data.numpy()
            clean_log_npy = clean.cpu().data.numpy()
            background_npy = background.cpu().data.numpy()
            Genh_log_npy = Genh.cpu().data.numpy()

            results_Noisy_log.append(noisy_log_npy[0])
            results_Clean_log.append(clean_log_npy[0])
            results_Genh_log.append(Genh_log_npy[0])
            results_Background.append(background_npy[0])

            # exp
            noisy_npy = noisy_log_npy[0] * (thetas_noisy.cpu().data.numpy()[0]) + min_value_noisy.cpu().data.numpy()[0]
            noisy_npy = np.exp(noisy_npy) - 1

            clean_npy = clean_log_npy[0] * (thetas_clean.cpu().data.numpy()[0]) + min_value_clean.cpu().data.numpy()[0]
            clean_npy = np.exp(clean_npy) - 1

            Genh_npy = Genh_log_npy[0] * (thetas_clean.cpu().data.numpy()[0]) + min_value_clean.cpu().data.numpy()[0]
            Genh_npy = np.exp(Genh_npy) - 1

            results_Noisy.append(noisy_npy)
            results_Clean.append(clean_npy)
            results_Genh.append(Genh_npy)

            # if ind == 1:
            #     plt.plot([i for i in range(2048)], noisy_npy, label='noisy')
            #     plt.plot([i for i in range(2048)], clean_npy, label='clean')
            #     plt.plot([i for i in range(2048)], Genh_npy, label='enhance')
            #     plt.legend(loc=0, ncol=3)
            #     # plt.show()
            #     plt.savefig(os.path.join(save_dir, f'energy_{ind}.png'))
            #     plt.close()
            #     # print(1)
        
        # pd.DataFrame(np.array(results)).to_csv(os.path.join(save_dir, 'enhance_result_ss.csv'))
        return np.array(results_Noisy), \
               np.array(results_Clean), \
               np.array(results_Genh), \
               np.array(results_Background), \
               np.array(results_Noisy_log), \
               np.array(results_Clean_log), \
               np.array(results_Genh_log)


    def train(self, 
              args: ArgumentParser, 
              dataloader: DataLoader, 
              criterion, 
              Goptimizer,
              Doptimizer,
              l1_weight, 
              l1_dec_step,
              l1_dec_epoch,
              writer: SummaryWriter = None,
              device=torch.device) -> None:
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.reg_loss = getattr(F, args.reg_loss)
        self.mse_loss = getattr(F, "mse_loss")
        iteration = 1
        z_sample = None

        label = torch.ones(args.batch_size)
        label = label.to(device)
        for epoch in range(1, args.epochs + 1):
            self.G.train()
            self.D.train()
            for ind, (noisy, clean, _, _, _, _) in enumerate(dataloader, start=1):
                if epoch >= l1_dec_epoch:
                    if l1_weight > 0:
                        l1_weight -= l1_dec_step
                        l1_weight = max(0, l1_weight) # ensure it is 0 if it goes < 0

                noisy = noisy.type(torch.FloatTensor).unsqueeze(1)
                clean = clean.type(torch.FloatTensor).unsqueeze(1)
                label.resize_(clean.size(0)).fill_(1)
                clean = clean.to(device)
                noisy = noisy.to(device)

                # (1) D real update
                Doptimizer.zero_grad()
                total_d_fake_loss = 0
                total_d_real_loss = 0
                Genh = self.infer_G(noisy, clean)
                lab = label
                d_real, _ = self.infer_D(clean, noisy)
                d_real_loss = criterion(d_real.view(-1), lab)
                d_real_loss.backward()
                total_d_real_loss += d_real_loss

                # (2) D fake update
                d_fake, _ = self.infer_D(Genh.detach(), noisy)
                lab = label.fill_(0)
                d_fake_loss = criterion(d_fake.view(-1), lab)
                d_fake_loss.backward()
                total_d_fake_loss += d_fake_loss
                Doptimizer.step()

                d_loss = d_fake_loss + d_real_loss 

                # (3) G real update
                Goptimizer.zero_grad()
                lab = label.fill_(1)
                d_fake_, _ = self.infer_D(Genh, noisy)
                g_adv_loss = criterion(d_fake_.view(-1), lab)
                #g_l1_loss = l1_weight * F.l1_loss(Genh, clean)
                g_l1_loss = l1_weight * self.reg_loss(Genh, clean)
                
                # local mse loss
                g_mse_loss = self.mse_loss(Genh, clean)
                for i in range(len(self.positions)):
                    g_mse_loss += self.mse_loss(Genh[:, :,self.positions[i]-30:self.positions[i]+30], 
                                                     clean[:, :, self.positions[i]-30:self.positions[i]+30])
                
                g_loss = g_adv_loss + g_l1_loss + 100 * g_mse_loss
                g_loss.backward()
                Goptimizer.step()

                if z_sample is None and not self.G.no_z:
                    # capture sample now that we know shape after first
                    # inference
                    z_sample = self.G.z[:20, :, :].contiguous()
                    print('z_sample size: ', z_sample.size())
                    z_sample = z_sample.to(device)
                if ind % 3 == 0 or ind >= len(dataloader):
                    d_real_loss_v = d_real_loss.cpu().item()
                    d_fake_loss_v = d_fake_loss.cpu().item()
                    g_adv_loss_v = g_adv_loss.cpu().item()
                    g_l1_loss_v = g_l1_loss.cpu().item()
                    g_mse_loss_v = g_mse_loss.cpu().item()

                    log = '(Iter {}) Batch {}/{} (Epoch {}) d_real:{:.4f}, d_fake:{:.4f}, '.format(iteration, ind, len(dataloader), epoch, d_real_loss_v, d_fake_loss_v)
                    log += 'g_adv:{:.4f}, g_l1:{:.4f}, g_local:{:.4f}, l1_w: {:.2f}, btime: {:.4f} s, mbtime: {:.4f} s'.format(g_adv_loss_v, g_l1_loss_v, g_mse_loss_v, l1_weight, 0, 0)
                    # print(log)

                    writer.add_scalar('D_real', d_real_loss_v, iteration)
                    writer.add_scalar('D_fake', d_fake_loss_v, iteration)
                    writer.add_scalar('G_adv', g_adv_loss_v, iteration)
                    writer.add_scalar('G_l1', g_l1_loss_v, iteration)
                    writer.add_scalar('G_local', g_mse_loss_v, iteration)
                    writer.add_histogram('D_fake__hist', d_fake_.cpu().data, iteration, bins='sturges')
                    writer.add_histogram('D_fake_hist', d_fake.cpu().data, iteration, bins='sturges')
                    writer.add_histogram('D_real_hist', d_real.cpu().data, iteration, bins='sturges')
                    writer.add_histogram('Gz', Genh.cpu().data, iteration, bins='sturges')
                    writer.add_histogram('clean', clean.cpu().data, iteration, bins='sturges')
                    writer.add_histogram('noisy', noisy.cpu().data, iteration, bins='sturges')
                    # get D and G weights and plot their norms by layer and
                    def model_weights_norm(model, total_name):
                        total_GW_norm = 0
                        for k, v in model.named_parameters():
                            if 'weight' in k:
                                W = v.data
                                W_norm = torch.norm(W)
                                writer.add_scalar('{}_Wnorm'.format(k), W_norm, iteration)
                                total_GW_norm += W_norm
                        writer.add_scalar('{}_Wnorm'.format(total_name), total_GW_norm, iteration)
                    model_weights_norm(self.G, 'Gtotal')
                    model_weights_norm(self.D, 'Dtotal')
                iteration += 1
                
            
            # save models in end of epoch with EOE savers
            if epoch % args.intervel == 0 or epoch >= args.epochs or epoch == 1:
                self.save(save_path=os.path.join(self.outputs, 'weights'),
                          step=epoch,
                          Goptimizer=Goptimizer, 
                          Doptimizer=Doptimizer)

    def save(self, 
             save_path: str, 
             step: int, 
             Goptimizer: Type[Union[optim.RMSprop, optim.Adam]], 
             Doptimizer: Type[Union[optim.RMSprop, optim.Adam]]) -> None:
        
        if not hasattr(self, 'save_path'):
            self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # save generator
        g_ckp = {'step': step,
                 'state_dict': self.G.state_dict()}
                 #'optimizer': Goptimizer.state_dict()}
        torch.save(g_ckp, os.path.join(save_path, f'g_{step}.pt'))
        
        # save discriminator
        # d_ckp = {'step': step,
        #          'state_dict': self.D.state_dict(),
        #          'optimizer': Doptimizer.state_dict()}
        # torch.save(d_ckp, os.path.join(save_path, f'd_{step}.pt'))
    
    def evaluate(self, dataloader, save_dir, do_noisy=False, max_samples=1, device='cpu'):
        r"""
            evaluate model and show the effect. 
        """

        self.G.eval()
        self.D.eval()

        results_Noisy_log = []
        results_Genh_log = []
        results_Clean_log = []

        results_Noisy = []
        results_Genh = []
        results_Clean = []
        results_Background = []
        
        # going over dataset ONCE
        for ind, (noisy, clean, thetas_noisy, thetas_clean, min_value_noisy, min_value_clean, background) in enumerate(dataloader, start=1):
            noisy = noisy.type(torch.FloatTensor).unsqueeze(1)
            
            clean = clean.to(device)
            noisy = noisy.to(device)
            background = background.to(device)
            Genh = self.infer_G(noisy).squeeze(1)

            noisy_log_npy = noisy.squeeze(1).cpu().data.numpy()
            clean_log_npy = clean.cpu().data.numpy()
            background_npy = background.cpu().data.numpy()
            Genh_log_npy = Genh.cpu().data.numpy()

            results_Noisy_log.append(noisy_log_npy[0])
            results_Clean_log.append(clean_log_npy[0])
            results_Genh_log.append(Genh_log_npy[0])
            results_Background.append(background_npy[0])

            # exp
            noisy_npy = noisy_log_npy[0] * (thetas_noisy.cpu().data.numpy()[0]) + min_value_noisy.cpu().data.numpy()[0]
            noisy_npy = np.exp(noisy_npy) - 1

            clean_npy = clean_log_npy[0] * (thetas_clean.cpu().data.numpy()[0]) + min_value_clean.cpu().data.numpy()[0]
            clean_npy = np.exp(clean_npy) - 1

            Genh_npy = Genh_log_npy[0] * (thetas_clean.cpu().data.numpy()[0]) + min_value_clean.cpu().data.numpy()[0]
            Genh_npy = np.exp(Genh_npy) - 1

            results_Noisy.append(noisy_npy)
            results_Clean.append(clean_npy)
            results_Genh.append(Genh_npy)

            # if ind == 1:
            #     plt.plot([i for i in range(2048)], noisy_npy, label='noisy')
            #     plt.plot([i for i in range(2048)], clean_npy, label='clean')
            #     plt.plot([i for i in range(2048)], Genh_npy, label='enhance')
            #     plt.legend(loc=0, ncol=3)
            #     # plt.show()
            #     plt.savefig(os.path.join(save_dir, f'energy_{ind}.png'))
            #     plt.close()
        
        # pd.DataFrame(np.array(results)).to_csv(os.path.join(save_dir, 'enhance_result_ss.csv'))
        return np.array(results_Noisy), \
               np.array(results_Clean), \
               np.array(results_Genh), \
               np.array(results_Background), \
               np.array(results_Noisy_log), \
               np.array(results_Clean_log), \
               np.array(results_Genh_log)