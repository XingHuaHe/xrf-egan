# Title: A new technique for baseline calibration of soil X-ray fluorescence spectrum based on enhanced generative adversarial networks combined with transfer learning


## Pre-training
### 1. If you are conducting pre-training experiments on a slurm cluster, you can execute the following code instructions to pre-train the model.
```
sbatch script_pretrain.sh
```
You can modify the relevant slurm cluster parameters in the script_pretrain.sh, including the path of the log output and the path of the python interpreter. Remember to set noisy and clean dataset file path in pretrain.py.

### 2. If you are not pre-training through the slurm cluster, you can pre-train the model by running the python file.
```
python3 pretrain.py \
--noisy-set="your pretraining noisy datas path (.mat)" \
--clean-set="your pretraining clean datas path (.mat)" \
--epochs=2000 \
--batch-size=2 \
--outputs="your output result saved path" \
--positions="the characteristic peak channel of the analyzed element, for example [156, 664, 303, 304, 288, 232, 215, 249]"
```

## Fine-tuning
### 1. If you are conducting fine-tuning experiments on a slurm cluster, you can execute the following code instructions to pre-train the model.
```
sbatch script_fine_tuning.sh
```
### 2. If you are not fine-tuning through the slurm cluster, you can fine-tuning the model by running the python file.
```
python3 finetune.py \
--noisy-set="your pretraining noisy datas path (.mat)" \
--clean-set="your pretraining clean datas path (.mat)" \
--G-pretrained-weight="generator weight file, saved in folder under the pre training output path (/`your project path`/outputs/`***`/pretrain)" \
--epochs=500 \
--batch-size=4 \
--outputs="your output result saved path" \
--positions="the characteristic peak channel of the analyzed element, for example [156, 664, 303, 304, 288, 232, 215, 249]"
```

## Leave-one-out cross-validation (LOOC)
### 1. If you want to do the LOOC experiment only once, use the following command
```
python3 loocv.py \
--noisy-set="your pretraining noisy datas path (.mat)" \
--clean-set="your pretraining clean datas path (.mat)" \
--G-pretrained-weight="generator weight file, saved in folder under the pre training output path (/`your project path`/outputs/`***`/pretrain)" \
--epochs=500 \
--batch-size=4 \
--outputs="your output result saved path" \
--positions="the characteristic peak channel of the analyzed element, for example [156, 664, 303, 304, 288, 232, 215, 249]"
```

### 2. If you want to run N times LOOC experiments to get the variance across multiple experiments, use the following command:
```
python3 loocv-n.py \
--n=5 \
--noisy-set="your pretraining noisy datas path (.mat)" \
--clean-set="your pretraining clean datas path (.mat)" \
--G-pretrained-weight="generator weight file, saved in folder under the pre training output path (/`your project path`/outputs/`***`/pretrain)" \
--epochs=500 \
--batch-size=4 \
--outputs="your output result saved path" \
--positions="the characteristic peak channel of the analyzed element, for example [156, 664, 303, 304, 288, 232, 215, 249]"
```

## Evaluate
If you have already trained the model, you can load the model, and perform validation analysis using the test dataset. The output results will be saved in the path specified by '--outputs' (/--outputs/test/).
```
python3 evaluate.py \
--noisy-set="your pretraining noisy datas path (.mat)" \
--clean-set="your pretraining clean datas path (.mat)" \
--G-pretrained-weight="generator weight file, saved in folder under the pre training output path (/`your project path`/outputs/`***`/pretrain)" \
--outputs="your output result saved path" \
```

## Analysis
The analysis code implemented through Matlab2021b are stored in the analysis folder, including $R^2$ and losses, etc.

## Results
![Experiment process](./imgs/img.png)

![Results](./imgs/Fig%207.%20Cu元素XRF基线校准。(a)XRF基线校准局部图.png)


## Citation
@article{
    title = {A new technique for baseline calibration of soil X-ray fluorescence spectrum based on enhanced generative adversarial networks combined with transfer learning},  
journal = { Journal of Analytical Atomic Spectrometry },  
year = { 28 Sep 2023 },  
doi = { https://doi.org/10.1039/D3JA00235G },  
author = { Xinghua He, Yanchun Zhao, Fusheng Li }
