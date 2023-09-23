import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd


def extract_datas(filepath):
    filenames = os.listdir(filepath)
    alloyDatas = []
    alloyId = []
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
            alloyId.append(Id)
            alloyDatas.append(energy)
    alloyDatas = np.array(alloyDatas)
    alloyId = np.array(alloyId)

    return alloyDatas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="/home/linsi/Projects/Energy GAN/datas/soil", help="data source")
    args = parser.parse_args()

    # Getting energy data.
    alloyDatas = extract_datas(args.source)

    print(alloyDatas[0])

