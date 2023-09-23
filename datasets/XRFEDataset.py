"""
针对 .dat 原始数据文件的 Dataset.
"""

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

class XRFEDataset_mat(Dataset):
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
            # train.
            self.noisy_energyDatas, self.thetas_noisy, self.min_value_noisy = self.extract_datas(noisy_path, training=self.training)
            self.clean_energyDatas, self.thetas_clean, self.min_value_clean = self.extract_clean_datas(clean_path, training=self.training)
        else:
            # evaluate.
            self.noisy_energyDatas, self.thetas_noisy, self.min_value_noisy = self.extract_datas(noisy_path, training=self.training)
            self.clean_energyDatas, self.thetas_clean, self.min_value_clean = self.extract_clean_datas(clean_path, training=self.training)
            self.backgrounds = self.extract_backgrounds(noisy_path, clean_path)

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
            filepath: file path.
            training: train (True), evaluate (False).
        """
        soilDatas = []
        thetas_noisy = []
        min_value_noisy = []

        matd = scio.loadmat(filepath)
        energys = matd['spectrums']

        for i in range(len(energys)):
            energy = energys[i]
            if training:
                energy, theta, mi = self.norm(energy)
                soilDatas.append(energy)
                thetas_noisy.append(theta)
                min_value_noisy.append(mi)
            else:
                energy, theta, mi = self.norm(energy)
                soilDatas.append(energy)
                thetas_noisy.append(theta)
                min_value_noisy.append(mi)
                    
        soilDatas = np.array(soilDatas)
        thetas_noisy = np.array(thetas_noisy)
        min_value_noisy = np.array(min_value_noisy)

        return soilDatas, thetas_noisy, min_value_noisy

    def extract_clean_datas(self, filepath: str, training: bool = True):
        r"""
            filepath: file path.
            training: train (True), evaluate (False).
        """

        soilCleans = []
        thetas_clean = []
        min_value_clean = []

        matd = scio.loadmat(filepath)
        energys = matd['spectrums'][:, 0:2048]

        for i in range(len(energys)):
            energy = energys[i]
            if training:
                energy, theta, mi = self.norm(energy) # y, theta, mi
                soilCleans.append(energy)
                thetas_clean.append(theta)
                min_value_clean.append(mi)
            else:
                # evaluate
                energy, theta, mi = self.norm(energy)
                thetas_clean.append(theta)
                soilCleans.append(energy)
                min_value_clean.append(mi)
   
        soilCleans = np.array(soilCleans)
        thetas_clean = np.array(thetas_clean)
        min_value_clean = np.array(min_value_clean)

        return soilCleans, thetas_clean, min_value_clean

        # dimension = 5
        # sigma = 1
        # kernel = np.zeros((1, dimension))
        # for loop in range(1, dimension + 1):
        #     x_axis = -(int)(dimension / 2) + loop - 1
        #     kernel[0, loop-1] = 1 / sigma * np.exp(-x_axis**2 / (2 * sigma ** 2))
        # kernel = kernel / np.sum(kernel)

        # for i in tqdm.tqdm(range(len(energys))):
        #     energy = energys[i]

        #     # 高斯卷积本底扣除, 如果向换另外一种本地扣除算法,可以在此处进行修改
        #     background = energy.copy()
        #     for i in range(500):
        #         new_energy = signal.convolve(background, kernel[0], mode='same') / sum(kernel[0])
        #         for i in range(2048):
        #             if new_energy[i] < background[i]:
        #                 background[i] = new_energy[i]
        #     for i in range(2048):
        #         energy[i] = energy[i] - background[i]

        #     soilCleans.append(energy)
    def extract_backgrounds(self, noisy_path: str, clean_path: str):
        

        noisy_matd = scio.loadmat(noisy_path)
        noisy = noisy_matd['spectrums'][:, 0:2048]

        clean_matd = scio.loadmat(clean_path)
        clean = clean_matd['spectrums'][:, 0:2048]

        backgrounds = noisy_matd - clean_matd

        return backgrounds
        

    def norm(self, x):
        r"""
        Normalization. 
        (1) When x[i] >= 1, y[i] = log x[i], and x[i] < 1, y[i] = 0.
        (2) (y[i] - min(y)) / (max(y) - min(y))
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



class XRFEDataset_dat(Dataset):
    """
        XRF enchance dataset for .dat files.
    """

    def __init__(self, noisy_path: str, clean_path: str = None, transforms: transforms = None, mode: bool = True) -> None:
        super().__init__()

        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.mode = mode
        self.transforms = transforms

        # extract xrf energy data.
        if mode:
            # train
            self.noisy_energyDatas, _ = self.extract_datas(noisy_path)
            if self.clean_path == None or self.clean_path == '':
                # 如果没有指定清洁的数据,那么利用噪声数据,利用高斯卷积本底扣除生成清洁数据
                self.clean_energyDatas, _ = self.extract_clean_datas(noisy_path)
            else:
                # 给定清洁的数据路径
                self.clean_energyDatas, _ = self.extract_datas(clean_path)
        else:
            # evaluate
            self.noisy_energyDatas, self.values = self.extract_datas(noisy_path)
            if self.clean_path == None or self.clean_path == '':
                self.clean_energyDatas, _ = self.extract_clean_datas(noisy_path, flag=True)
            else:
                self.clean_energyDatas, _ = self.extract_datas(clean_path, flag=True)

    def __getitem__(self, index):
        if self.mode:
            noisy = self.noisy_energyDatas[index]
            clean = self.clean_energyDatas[index]
            return noisy, clean
        else:
            noisy = self.noisy_energyDatas[index]
            clean = self.clean_energyDatas[index]
            value = self.values[index]
            return noisy, clean, value

    
    def __len__(self) -> int:
        return len(self.noisy_energyDatas)

    def extract_datas(self, filepath: str, flag: bool = False):

        filenames = os.listdir(filepath)
        soilDatas = []
        thetas = []
        for filename in filenames:
            if filename.split('.')[-1].lower() == 'xls':
                # labels = pd.read_excel(os.path.join(filepath, filename))
                continue
            elif filename.split('.')[-1].lower() != 'dat':
                continue
            else:
                # extract .dat.
                with open(os.path.join(filepath, filename), 'rb') as f:
                    data = f.read()
                    energy = []
                    m = 5
                    for _ in range(2048):
                        temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 30
                        energy.append(temp)
                        m += 3
                    Id = data[3] * 256 + data[4]
                if Id < -1:
                    continue
                
                if not flag:
                    energy, theta = self.norm(energy)
                    thetas.append(theta)
                    soilDatas.append(energy)
                else:
                    _, theta = self.norm(energy)
                    thetas.append(theta)
                    soilDatas.append(energy)
                    
        soilDatas = np.array(soilDatas)
        thetas = np.array(thetas)

        return soilDatas, thetas

    def extract_clean_datas(self, filepath: str, flag: bool = False):
        print("Clean set is not given. Extracting clean energy datas")

        filenames = os.listdir(filepath)
        soilDatas = []
        thetas = []

        dimension = 5
        sigma = 1
        kernel = np.zeros((1, dimension))
        for loop in range(1, dimension + 1):
            x_axis = -(int)(dimension / 2) + loop - 1
            kernel[0, loop-1] = 1 / sigma * np.exp(-x_axis**2 / (2 * sigma ** 2))
        kernel = kernel / np.sum(kernel)

        for filename in tqdm.tqdm(filenames):
            if filename.split('.')[-1].lower() == 'xls':
                # labels = pd.read_excel(os.path.join(filepath, filename))
                continue
            elif filename.split('.')[-1].lower() != 'dat':
                continue
            else:
                # extract .dat.
                with open(os.path.join(filepath, filename), 'rb') as f:
                    data = f.read()
                    energy = []
                    m = 5
                    for _ in range(2048):
                        temp = (data[m] * 256 * 256 + data[m+1] * 256 + data[m+2]) / 30
                        energy.append(temp)
                        m += 3
                    Id = data[3] * 256 + data[4]
                if Id < -1:
                    continue

                # 高斯卷积本底扣除, 如果向换另外一种本地扣除算法,可以在此处进行修改
                background = energy.copy()
                for i in range(1000):
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

        return soilDatas, thetas

    def norm(self, x):
        r"""
        Normalization. 
        (1) When x[i] >= 1, y[i] = log x[i], and x[i] < 1, y[i] = 0.
        (2) y[i] / (max(y) - min(y))
        """
        y = []
        for i in range(len(x)):
            if x[i] >= 1:
                y.append(np.log(x[i]))
            else:
                y.append(0)
        mi = min(y)
        ma = max(y)
        theta = (ma - mi)
        y = y / theta
        return y, theta


# if __name__ == "__main__":
#     xrfDataset = XRFEDataset("/home/linsi/Projects/Energy GAN/datas/soil")
#     from torch.utils.data import DataLoader
#     xrfDataloader = DataLoader(xrfDataset, 1,True)

#     for i, item in enumerate(xrfDataloader):
#         print(item)
