#！/bin/bash

#edqn
load_dir="/FactoryTaskAllocationMiC_2024-12-11_13-29-49/nn"
train=
relative_pth="/omniisaacgymenvs/runs"
str="/"
work_space_path=$(pwd)
dir_path=$work_space_path$relative_pth$load_dir
# path=$1
files=$(ls $dir_path)

# for filename in $files
# do
# for num in $list
# do
#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
#     load_dir="$load_dir" load_name="$str$filename" wandb_project=test_zero_shot test_times=10 num_product=$num
# #    echo $filename >> filename.txt
# #    echo -e >> filename.txt
# done
# done




# python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2025-05-24_13-25-03/nn" load_name=/FactoryTaskAllocationMiC_ep_20500.pth wandb_project=test_move test_times=10 

python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2025-05-25_11-32-15/nn" load_name=/FactoryTaskAllocationMiC_ep_21900.pth wandb_project=test_move test_times=3
