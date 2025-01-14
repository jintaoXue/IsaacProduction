#！/bin/bash
# 读取不同的学习率
load_dir="/FactoryTaskAllocationMiC_2024-12-09_21-42-46/nn"
relative_pth="/omniisaacgymenvs/runs"
str="/"
work_space_path=$(pwd)
dir_path=$work_space_path$relative_pth$load_dir
# path=$1
files=$(ls $dir_path)

# for f in "$dir_path"/*; do
#   echo $f >> filename.txt
#   echo -e >> filename.txt
# done

# for filename in $files
# do
#    echo $filename >> filename.txt
# #    echo -e >> filename.txt
# done

# for filename in $files
# do
#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbownoe headless=True wandb_activate=True test=True \
#     load_dir="$load_dir" load_name="$str$filename" wandb_project=test_HRTA test_times=100 
# #    echo $filename >> filename.txt
# #    echo -e >> filename.txt
# done





for num in {1..10}
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-23_18-12-29/nn" load_name=/FactoryTaskAllocationMiC_ep_20100.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num

    # python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=edqn headless=True wandb_activate=True test=True \
    # load_dir="/FactoryTaskAllocationMiC_2024-12-11_13-29-49/nn" load_name=/FactoryTaskAllocationMiC_ep_10500.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num

    # python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCepsilon_noisy headless=True wandb_activate=True test=True \
    # load_dir="/FactoryTaskAllocationMiC_2024-12-09_14-31-02/nn" load_name="/FactoryTaskAllocationMiC_ep_5700.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num

    # python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=no_dueling headless=True wandb_activate=True test=True \
    # load_dir="/FactoryTaskAllocationMiC_2024-12-10_13-06-23/nn" load_name="/FactoryTaskAllocationMiC_ep_24900.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num

    # python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
    # load_dir="/FactoryTaskAllocationMiC_2024-12-08_15-44-10/nn" load_name="/FactoryTaskAllocationMiC_ep_24000.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num

done


python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-23_18-12-29/nn" load_name=/FactoryTaskAllocationMiC_ep_20100.pth wandb_project=test_HRTA test_times=100

