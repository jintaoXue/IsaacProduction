#！/bin/bash
# 读取不同的学习率
# load_dir="/FactoryTaskAllocationMiC_2024-12-08_17-36-58/nn"
load_dir=$1
load_name="/FactoryTaskAllocationMiC_ep_25000.pth"
train=$2
relative_pth="/omniisaacgymenvs/runs"
str="/"
work_space_path=$(pwd)
dir_path=$work_space_path$relative_pth$load_dir
# path=$1
files=$(ls $dir_path)
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

for filename in $files
do
for num in $list
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=$train headless=True wandb_activate=True test=True \
    load_dir="$load_dir" load_name="$str$filename" wandb_project=test_zero_shot test_times=10 num_product=$num
#    echo $filename >> filename.txt
#    echo -e >> filename.txt
done
done



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
