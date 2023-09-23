#!/bin/bash
#SBATCH -J exp-2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -D /home/"your python interpreter path"
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -o /home/"your project path"/exp-2.log
#SBATCH -e /home/"your project path"/exp-2.err


echo 'Fine tuning XRFEGAN model!'
echo "Start Time : `date`"


CUDA_VISIBLE_DEVICES=0 /`your python interpreter path`/bin/python /home/"your project path"/finetune.py --noisy-set="noisy.mat" --clean-set="clean.mat"
echo "End Time : `date`"
