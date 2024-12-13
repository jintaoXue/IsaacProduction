#！/bin/bash
# 读取不同的学习率


batch_size_list=(
    128
    256
    512
)

train="FactoryTaskAllocationMiCRainbowepsilon"



for size in $batch_size_list
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=$train headless=True wandb_activate=True \
    load_dir="$load_dir" load_name="$str$filename" wandb_project=test_zero_shot test_times=10 num_product=$num
#    echo $filename >> filename.txt
#    echo -e >> filename.txt
done
done
