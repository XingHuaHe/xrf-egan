import numpy as np
import scipy.io as scio
from scipy import signal
from torch.utils.data import Dataset
from tqdm import tqdm

def norm(x):
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


def extract_datas(noisy_set: str, clean_set: str):
    r"""
        filepath: file path.
        training: train (True), evaluate (False).
    """
    xrfDatas = []
    xrfDatas_n = [] 
    xrfCleanDatas = []
    xrfCleanDatas_n = []
    thetas_noisy = []
    thetas_clean = []
    min_value_noisy = []
    min_value_clean = []
    backgrounds = []

    noisy_matd = scio.loadmat(noisy_set)
    noisy_energys = noisy_matd['spectrums'][:, 0:2048]
    clean_matd = scio.loadmat(clean_set)
    clean_energys = clean_matd['spectrums'][:, 0:2048]

    for i in tqdm(range(len(noisy_energys))):
        noisy = noisy_energys[i]
        clean = clean_energys[i]

        background = noisy - clean

        energy_n, theta_noisy, mi_noisy = norm(noisy)
        energy_clean_n, theta_clean, mi_clean = norm(clean)

        xrfDatas.append(noisy)
        xrfDatas_n.append(energy_n)
        xrfCleanDatas.append(clean)
        xrfCleanDatas_n.append(energy_clean_n)
        thetas_noisy.append(theta_noisy)
        thetas_clean.append(theta_clean)
        min_value_clean.append(mi_noisy)
        min_value_noisy.append(mi_clean)
        backgrounds.append(background)
                
    xrfDatas = np.array(xrfDatas)
    xrfDatas_n = np.array(xrfDatas_n)
    xrfCleanDatas = np.array(xrfCleanDatas)
    xrfCleanDatas_n = np.array(xrfCleanDatas_n) 
    thetas_noisy = np.array(thetas_noisy) # 
    thetas_clean = np.array(thetas_clean)
    min_value_clean = np.array(min_value_clean) #
    min_value_noisy = np.array(min_value_noisy) #
    backgrounds = np.array(backgrounds)

    return xrfDatas, xrfDatas_n, xrfCleanDatas, xrfCleanDatas_n, thetas_noisy, thetas_clean, min_value_noisy, min_value_clean, backgrounds


class LOOCVDataset(Dataset):
    def __init__(self, noisy, clean, thetas_noisy=None, thetas_clean=None, min_value_noisy=None, min_value_clean=None, backgrounds=None, training: bool = True) -> None:
        super().__init__()

        self.training = training
        if self.training:
            self.noisy_energyDatas = noisy
            self.clean_energyDatas = clean
            self.thetas_noisy = thetas_noisy
            self.thetas_clean = thetas_clean
            self.min_value_noisy = min_value_noisy
            self.min_value_clean = min_value_clean
            self.backgrounds = backgrounds
        else:
            self.noisy_energyDatas = np.array([noisy])
            self.clean_energyDatas = np.array([clean])
            self.thetas_noisy = np.array([thetas_noisy])
            self.thetas_clean = np.array([thetas_clean])
            self.min_value_noisy = np.array([min_value_noisy])
            self.min_value_clean = np.array([min_value_clean])
            self.backgrounds = np.array([backgrounds])

    def __getitem__(self, index):
        if self.training:
            noisy = self.noisy_energyDatas[index]
            clean = self.clean_energyDatas[index]
            thetas_noisy = self.thetas_noisy[index]
            thetas_clean = self.thetas_clean[index]
            min_value_noisy = self.min_value_noisy[index]
            min_value_clean = self.min_value_clean[index]
            return noisy, clean, thetas_noisy, thetas_clean, min_value_noisy, min_value_clean
        else:
            noisy = self.noisy_energyDatas[index]
            clean = self.clean_energyDatas[index]
            thetas_noisy = self.thetas_noisy[index]
            thetas_clean = self.thetas_clean[index]
            min_value_noisy = self.min_value_noisy[index]
            min_value_clean = self.min_value_clean[index]
            background = self.backgrounds[index]
            
            return noisy, clean, thetas_noisy, thetas_clean, min_value_noisy, min_value_clean, background

    
    def __len__(self) -> int:
        return len(self.noisy_energyDatas)