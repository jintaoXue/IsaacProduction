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

######long
# python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2025-05-25_11-32-15/nn" load_name=/FactoryTaskAllocationMiC_ep_21900.pth wandb_project=test_move test_times=100

######random
# python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2024-12-23_18-12-29/nn" load_name=/FactoryTaskAllocationMiC_ep_20100.pth wandb_project=test_move test_times=100



############### short new

list=(
 52900
 51800
 50000
 49300
 42000
 39600
)

for num in "${list[@]}"
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
        load_dir="/FactoryTaskAllocationMiC_2025-05-28_22-16-18/nn" load_name=/FactoryTaskAllocationMiC_ep_$num.pth wandb_project=test_move test_times=100
    # echo $num
#    echo -e >> filename.txt
done




#####short

# list=(
#     24600
#     23600
#     23400
#     23300
#     23100
#     22900
#     22600
#     22200
#     21800
#     # 21100
#     21000
#     20800
#     20700
#     20600
#     20500
#     20300
#     20000
#     19800
#     19500
#     19300
#     19000
#     18900
#     18800
#     18700
#     18600
#     18500
#     18200
#     17500
#     17400
#     16900
#     16600
#     16200
#     15500
#     15300
# )

# for num in "${list[@]}"
# do
#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
#         load_dir="/FactoryTaskAllocationMiC_2025-05-24_13-25-03/nn" load_name=/FactoryTaskAllocationMiC_ep_$num.pth wandb_project=test_move test_times=100
#     # echo $num
# #    echo -e >> filename.txt
# done

