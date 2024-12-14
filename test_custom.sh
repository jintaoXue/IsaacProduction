#！/bin/bash
# 读取不同的学习率

#edqn
load_dir="/FactoryTaskAllocationMiC_2024-12-11_13-29-49/nn"
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
for filename in $files
do
for num in $list
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=edqn headless=True wandb_activate=True test=True \
    load_dir="$load_dir" load_name="$str$filename" wandb_project=test_zero_shot test_times=10 num_product=$num
#    echo $filename >> filename.txt
#    echo -e >> filename.txt
done
done


