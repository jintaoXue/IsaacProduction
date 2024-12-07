#！/bin/bash
# 读取不同的学习率


declare -a arr=("FactoryTaskAllocationMiCRainbowmini" "FactoryTaskAllocationMiCRainbownoe")


for i in "${arr[@]}"
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train="$i" headless=True wandb_activate=True
done
