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
list=(
    1
    2
    3
    4
    5
)

# files=$(ls $dir_path)

# for f in "$dir_path"/*; do
#   echo $f >> filename.txt
#   echo -e >> filename.txt
# done

# for filename in $files
# do
# for num in $list
# do
#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=$train headless=True wandb_activate=True test=True \
#     load_dir="$load_dir" load_name="$str$filename" wandb_project=test_zero_shot test_times=10 num_product=$num
# #    echo $filename >> filename.txt
# #    echo -e >> filename.txt
# done
# done


# for num in $list
# do
#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=edqn headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2024-12-11_13-29-49/nn" load_name=/FactoryTaskAllocationMiC_ep_10500.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num

#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCepsilon_noisy headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2024-12-09_14-31-02/nn" load_name="/FactoryTaskAllocationMiC_ep_5700.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num

#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=no_dueling headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2024-12-10_13-06-23/nn" load_name="/FactoryTaskAllocationMiC_ep_24900.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num

#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini headless=True wandb_activate=True test=True \
#     load_dir="/FactoryTaskAllocationMiC_2024-12-08_15-44-10/nn" load_name="/FactoryTaskAllocationMiC_ep_24000.pth" wandb_project="test_zero_shot$num" test_times=10 num_product=$num
# #    echo $filename >> filename.txt
# #    echo -e >> filename.txt
# done


for num in $list
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbownoe headless=True wandb_activate=True test=True \
    load_dir="/FactoryTaskAllocationMiC_2024-12-09_21-42-46/nn" load_name=/FactoryTaskAllocationMiC_ep_6100.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num

#    echo $filename >> filename.txt
#    echo -e >> filename.txt
done


# for filename3 in $files3
# do
# for num in $list
# do
#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=$train3 headless=True wandb_activate=True test=True \
#     load_dir="$load_dir3" load_name="$str$filename3" wandb_project=test_zero_shot test_times=10 num_product=$num
# #    echo $filename >> filename.txt
# #    echo -e >> filename.txt
# done
# done


# for num in $list
# do
#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowepsilon headless=True wandb_activate=True test=True \
#     load_dir="$load_dir" load_name="$load_name" wandb_project=test_zero_shot test_times=10 num_product=$num
# #    echo $filename >> filename.txt
# #    echo -e >> filename.txt
# done



# # 循环执行命令
# for lr in "${lrs[@]}";do
#     python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCRainbowmini  headless=True wandb_activate=False
# 	python ./src/run.py --model DNN --data data/normalized/EdgeIIoT_normalized.csv --epoch 1 --lr "$lr" --method EDL --ood False --ensemble True --ks False >> ./src/output.txt
# done
