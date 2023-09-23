# Title: A new technique for baseline calibration of soil X-ray fluorescence spectrum based on enhanced generative adversarial networks combined with transfer learning


## 1. 预训练
1.如果你是在 slurm 集群进行实验，可以执行以下代码指令进行模型预训练
```
sbatch script_pretrain.sh
```
你可以修改 script_pretrain.sh 脚本文件中的相关参数，包括日志输出的路径，python interpreter 的路径. 记得设置noisy and clean dataset file path.

2.如果你不是通过 slurm 集群服务器进行训练，你可以通过运行 python 文件进行模型预训练
```
python3 pretrain.py \
--noisy-set="your pretraining noisy datas path (.mat)" \
--clean-set="your pretraining clean datas path (.mat)" \
--epochs=2000 \
--batch-size=2 \
--outputs="your output result saved path" \
--positions="the characteristic peak channel of the analyzed element, for example [156, 664, 303, 304, 288, 232, 215, 249]"
```

## 2. 微调
1.如果你是在 slurm 集群进行实验，可以执行以下代码指令进行模型预训练
```
sbatch script_fine_tuning.sh
```
2.如果你不是通过 slurm 集群服务器进行训练，你可以通过运行 python 文件进行模型预训练
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

## 3. 留一交叉验证
1.如果你想只执行一次实验，通过交叉验证，使用以下指令
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

2.如果您想进行 n 次交叉验证实验，以获得多次实验的方差，使用以下指令：
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


## 4. Evaluate

![流程](./imgs/img.png)