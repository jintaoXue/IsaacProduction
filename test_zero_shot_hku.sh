#！/bin/bash
# 读取不同的学习率
load_dir="/FactoryTaskAllocationMiC_2024-12-08_15-44-10/nn"
load_dir2="/FactoryTaskAllocationMiC_2024-12-10_13-06-23/nn"
# load_dir3="/FactoryTaskAllocationMiC_2024-12-08_15-44-10/nn"
# load_name="/FactoryTaskAllocationMiC_ep_25000.pth"
train=FactoryTaskAllocationMiCRainbowmini
train2=no_dueling
# train3=FactoryTaskAllocationMiCRainbowmini

relative_pth="/omniisaacgymenvs/runs"
str="/"
work_space_path=$(pwd)
dir_path=$work_space_path$relative_pth$load_dir
dir_path2=$work_space_path$relative_pth$load_dir2
# dir_path3=$work_space_path$relative_pth$load_dir3
# path=$1
files=$(ls $dir_path)
files2=$(ls $dir_path2)
# files3=$(ls $dir_path3)




# for num in {3..4}
# do
#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=edqn headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2024-12-11_13-29-49/nn" load_name=/FactoryTaskAllocationMiC_ep_10500.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num

#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCepsilon_noisy headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2024-12-09_14-31-02/nn" load_name="/FactoryTaskAllocationMiC_ep_5700.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num

#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=no_dueling headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2024-12-10_13-06-23/nn" load_name="/FactoryTaskAllocationMiC_ep_24900.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num

#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2024-12-08_15-44-10/nn" load_name="/FactoryTaskAllocationMiC_ep_24000.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num

# done



for num in {5..5}
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=edqn headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-11_13-29-49/nn" load_name=/FactoryTaskAllocationMiC_ep_10500.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2000

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCepsilon_noisy headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-09_14-31-02/nn" load_name="/FactoryTaskAllocationMiC_ep_5700.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2000

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=no_dueling headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-10_13-06-23/nn" load_name="/FactoryTaskAllocationMiC_ep_24900.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2000

done

for num in {6..8}
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=edqn headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-11_13-29-49/nn" load_name=/FactoryTaskAllocationMiC_ep_10500.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2500

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCepsilon_noisy headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-09_14-31-02/nn" load_name="/FactoryTaskAllocationMiC_ep_5700.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2500

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=no_dueling headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-10_13-06-23/nn" load_name="/FactoryTaskAllocationMiC_ep_24900.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2500

done

for num in {9..10}
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=edqn headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-11_13-29-49/nn" load_name=/FactoryTaskAllocationMiC_ep_10500.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=3000

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCepsilon_noisy headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-09_14-31-02/nn" load_name="/FactoryTaskAllocationMiC_ep_5700.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=3000

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=no_dueling headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-10_13-06-23/nn" load_name="/FactoryTaskAllocationMiC_ep_24900.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=3000

done