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



for num in {5..5}
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC headless=True wandb_activate=True test=True \
    train=FactoryTaskAllocationMiCRainbownoe load_dir="/FactoryTaskAllocationMiC_2024-12-09_21-42-46/nn" load_name=/FactoryTaskAllocationMiC_ep_8000.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2000

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC headless=True wandb_activate=True test=True \
    train=FactoryTaskAllocationMiCRainbowepsilon load_dir="/FactoryTaskAllocationMiC_2024-12-08_17-36-58/nn" load_name=/FactoryTaskAllocationMiC_ep_19500.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2000

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC headless=True wandb_activate=True test=True \
    train=FactoryTaskAllocationMiCRainbowmini load_dir="/FactoryTaskAllocationMiC_2024-12-08_15-44-10/nn" load_name=/FactoryTaskAllocationMiC_ep_24000.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2000
done

for num in {6..8}
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC headless=True wandb_activate=True test=True \
    train=FactoryTaskAllocationMiCRainbownoe load_dir="/FactoryTaskAllocationMiC_2024-12-09_21-42-46/nn" load_name=/FactoryTaskAllocationMiC_ep_8000.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2500

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC headless=True wandb_activate=True test=True \
    train=FactoryTaskAllocationMiCRainbowepsilon load_dir="/FactoryTaskAllocationMiC_2024-12-08_17-36-58/nn" load_name=/FactoryTaskAllocationMiC_ep_19500.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2500

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC headless=True wandb_activate=True test=True \
    train=FactoryTaskAllocationMiCRainbowmini load_dir="/FactoryTaskAllocationMiC_2024-12-08_15-44-10/nn" load_name=/FactoryTaskAllocationMiC_ep_24000.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=2500
done

for num in {9..10}
do
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC headless=True wandb_activate=True test=True \
    train=FactoryTaskAllocationMiCRainbownoe load_dir="/FactoryTaskAllocationMiC_2024-12-09_21-42-46/nn" load_name=/FactoryTaskAllocationMiC_ep_8000.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=3000

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC headless=True wandb_activate=True test=True \
    train=FactoryTaskAllocationMiCRainbowepsilon load_dir="/FactoryTaskAllocationMiC_2024-12-08_17-36-58/nn" load_name=/FactoryTaskAllocationMiC_ep_19500.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=3000

    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC headless=True wandb_activate=True test=True \
    train=FactoryTaskAllocationMiCRainbowmini load_dir="/FactoryTaskAllocationMiC_2024-12-08_15-44-10/nn" load_name=/FactoryTaskAllocationMiC_ep_24000.pth wandb_project="test_zero_shot$num" test_times=10 num_product=$num test_env_max_length=3000
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
