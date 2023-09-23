from json import load
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import tqdm
# from utils.energyExtract import extract_datas
import scipy.io as scio

class SimulationXRFEDataset_mat(Dataset):
    """
        XRF enchance dataset for .mat files.
    """
    def __init__(self, noisy_path: str, clean_path: str = None, transforms: transforms = None, training: bool = True) -> None:
        super().__init__()

        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.training = training
        self.transforms = transforms

        if self.training:
            # train
            self.noisy_energyDatas, self.thetas_noisy, self.thetas_clean, self.min_value_noisy, self.min_value_clean, self.backgrounds, self.clean_energyDatas = self.extract_datas(noisy_path, training=self.training)

        else:
            # evaluate
            self.noisy_energyDatas, self.thetas_noisy, self.thetas_clean, self.min_value_noisy, self.min_value_clean, self.backgrounds, self.clean_energyDatas = self.extract_datas(noisy_path, training=self.training)

    def __getitem__(self, index):
        if self.training:
            noisy = self.noisy_energyDatas[index]
            thetas_noisy = self.thetas_noisy[index]
            min_value_noisy = self.min_value_noisy[index]

            clean = self.clean_energyDatas[index]
            thetas_clean = self.thetas_clean[index]
            min_value_clean = self.min_value_clean[index]

            return noisy, clean, thetas_noisy, thetas_clean, min_value_noisy, min_value_clean
        else:
            noisy = self.noisy_energyDatas[index]
            thetas_noisy = self.thetas_noisy[index]
            min_value_noisy = self.min_value_noisy[index]

            clean = self.clean_energyDatas[index]
            thetas_clean = self.thetas_clean[index]
            min_value_clean = self.min_value_clean[index]

            background = self.backgrounds[index]
            return noisy, clean, thetas_noisy, thetas_clean, min_value_noisy, min_value_clean, background

    
    def __len__(self) -> int:
        return len(self.noisy_energyDatas)
        
    def extract_datas(self, filepath: str, training: bool = True):
        r"""
        filepath    : 提取的文件路径
        training    : 训练还是测试，evaluate (False)
        """
        soilDatas = []
        thetas_noisy = []
        thetas_clean = []
        min_value_noisy = []
        min_value_clean = []
        soilCleans = []

        matd = scio.loadmat(filepath)
        energys = matd['spectrums'][:, 0:2048]
        backgrounds = matd['spectrums'][:, 2048:]

        cleans = energys - backgrounds

        for i in range(len(energys)):
            energy = energys[i]
            clean = cleans[i]
            if training:
                energy, theta, mi = self.norm(energy)
                thetas_noisy.append(theta)
                min_value_noisy.append(mi)
                soilDatas.append(energy)

                clean, theta, mi = self.norm(clean)
                thetas_clean.append(theta)
                min_value_clean.append(mi)
                soilCleans.append(clean)
            else:
                energy, theta, mi = self.norm(energy)
                thetas_noisy.append(theta)
                min_value_noisy.append(mi)
                soilDatas.append(energy)

                clean, theta, mi = self.norm(clean)
                thetas_clean.append(theta)
                min_value_clean.append(mi)     
                soilCleans.append(clean)
                    
        soilDatas = np.array(soilDatas)
        thetas_noisy = np.array(thetas_noisy)
        thetas_clean = np.array(thetas_clean)
        min_value_noisy = np.array(min_value_noisy)
        min_value_clean = np.array(min_value_clean)
        soilCleans = np.array(soilCleans)

        return soilDatas, thetas_noisy, thetas_clean, min_value_noisy, min_value_clean, backgrounds, soilCleans

    def extract_clean_datas(self, filepath: str, flag: bool = False):
        print("Clean set is not given. Extracting clean energy datas")

        soilDatas = []
        thetas = []
        matd = scio.loadmat(filepath)
        energys = matd['spectrums'][:, 0:2048]
        contents = matd['spectrums'][:, 2048:]

        dimension = 5
        sigma = 1
        kernel = np.zeros((1, dimension))
        for loop in range(1, dimension + 1):
            x_axis = -(int)(dimension / 2) + loop - 1
            kernel[0, loop-1] = 1 / sigma * np.exp(-x_axis**2 / (2 * sigma ** 2))
        kernel = kernel / np.sum(kernel)

        for i in tqdm.tqdm(range(len(energys))):
            energy = energys[i]

            # 高斯卷积本底扣除, 如果向换另外一种本地扣除算法,可以在此处进行修改
            background = energy.copy()
            for i in range(500):
                new_energy = signal.convolve(background, kernel[0], mode='same') / sum(kernel[0])
                for i in range(2048):
                    if new_energy[i] < background[i]:
                        background[i] = new_energy[i]
            for i in range(2048):
                energy[i] = energy[i] - background[i]

            if not flag:
                energy, theta = self.norm(energy)
                thetas.append(theta)
                soilDatas.append(energy)
            else:
                # clean.
                _, theta = self.norm(energy)
                thetas.append(theta)
                soilDatas.append(energy)
                    
        soilDatas = np.array(soilDatas)
        thetas = np.array(thetas)

        print("Finished extract energy datas")

        # soilDatas = np.load('./synthetic spectrum/soilDatas.npy')
        # thetas = np.load('./synthetic spectrum/thetas.npy')

        return soilDatas, thetas

    def norm(self, x):
        r"""
        Normalization. 
        (1) When x[i] >= 1, y[i] = log x[i], and x[i] < 1, y[i] = 0.
        (2) y[i] / (max(y) - min(y))
        """
        y = []
        for i in range(len(x)):
            y.append(np.log(x[i]+1))
            # if x[i] >= 1:
            #     y.append(np.log(x[i]+1))
            # else:
            #     y.append(np.log(1)) # np.log(1) == 0
        mi = min(y)
        ma = max(y)
        theta = (ma - mi)
        y = (y - mi) / theta
        return y, theta, mi