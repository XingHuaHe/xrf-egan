#!/bin/bash
#SBATCH -J exp-4
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -D /home/yczhao/xinghua.he/test
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -o /home/yczhao/xinghua.he/test/exp-4.log
#SBATCH -e /home/yczhao/xinghua.he/test/exp-4.err


echo 'Train XRFEGAN model by LOOCV for N!'
echo "Start Time : `date`"


CUDA_VISIBLE_DEVICES=0 /`your python interpreter path`/bin/python /home/"your project path"/loocv-n.py --n=5 --noisy-set="noisy.mat" --clean-set="clean.mat"

echo "End Time : `date`"
