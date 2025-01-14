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


#mean + zeroshot for no spatial
python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-23_18-12-29/nn" load_name=/FactoryTaskAllocationMiC_ep_20100.pth wandb_project=test_HRTA test_times=100 


for num in {1..10}
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-23_18-12-29/nn" load_name=/FactoryTaskAllocationMiC_ep_20100.pth wandb_project=test_zero_shot test_times=10 num_product=$num
#    echo $filename >> filename.txt
#    echo -e >> filename.txt
done

