#！/bin/bash
# 读取不同的学习率


batch_size_list=(
    128
    256
)

train="FactoryTaskAllocationMiCRainbowepsilon"



for size in $batch_size_list
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=$train headless=True wandb_activate=True \
        batch_size=$size
#    echo $filename >> filename.txt
#    echo -e >> filename.txt
done
done
