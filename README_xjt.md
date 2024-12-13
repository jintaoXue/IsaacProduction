# Objects info in task allocation 
Dataset/3D_model/Part1/1_1.usd
是上料管道的运输架



omniisaacgymenvs/cfg/task/FactoryEnvTaskAllocation.yaml
omniisaacgymenvs/cfg/task/FactoryTaskAllocation.yaml
omniisaacgymenvs/tasks/factory/yaml/factory_asset_info_task_allocation.yaml
omniisaacgymenvs/cfg/train/FactoryTaskAllocationPPO.yaml

# 进度
issac-sim 中旋转 90 90 -90

# 下一步尝试用代码控制物体
core API:
https://docs.omniverse.nvidia.com/py/isaacsim/index.html
https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_omnigraph_shortcuts.html?highlight=control%20graph#controller-graphs controller
 

Part1 上料输送轨道+激光除锈工位+下料输送轨道
Part2 法兰料架x2，左+右
Part3 固定物体
Part4 龙门架静态
Part5 运料框
Part6 固定物体 气罐
Part7 龙门架，下料抓手
Part8 龙门架中间放料区
Part9 上料抓手 
Part10 激光切料区
Part11 激光焊接区



Part10 激光切料区
    '''reset laser cutter'''
    dof_pos_10 = torch.tensor([[0., 0]], device='cuda:0')
    dof_vel_10 = torch.tensor([[0., 0]], device='cuda:0')
    self.obj_part_10.set_joint_positions(dof_pos_10[0])
    self.obj_part_10.set_joint_velocities(dof_vel_10[0])
    '''start laser cutter'''
    dof_pos_10 = torch.tensor([[-5, 0.35]], device='cuda:0')
    dof_vel_10 = torch.tensor([[0., 0]], device='cuda:0')
    self.obj_part_10.set_joint_positions(dof_pos_10[0])
    self.obj_part_10.set_joint_velocities(dof_vel_10[0])

#7.12
还差什么部分：
self.station_state_inner_left = 4结束之后怎么复原

如果inner station 发出welding upper 的请求，先确认inner gripper是resetting 状态
prim_transformation:
https://docs.omniverse.nvidia.com/isaacsim/latest/how_to_guides/environment_setup.html?highlight=transform%2520matrix#adding-a-transform-matrix-to-a-prim

#7.14 local_pose transformation to global

https://docs.omniverse.nvidia.com/isaacsim/latest/features/warehouse_logistics/ext_omni_anim_people.html


# character_prim_path可能需要注意
question: https://forums.developer.nvidia.com/t/how-to-control-peoples-animation-in-isaac-using-python-api-rather-than-the-omni-anim-people-extension-with-ui/301824

# 8.6

route planning: https://developer.nvidia.com/blog/optimizing-robot-route-planning-with-nvidia-cuopt-for-isaac-sim/

# 8.12
occupancy map
x-50 30
y-30 40

# 8.13
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

# 8.24
path planner需要继续优化

cutting cube 的操作用worker去做，可以尝试判断一下位姿关系，选择靠近cutting machine的worker
worker 0 要呆在左边

# 8.25
基本搞定 product的放置位置还需要调整 以及大循环的终止条件还需要修改

# 8.26
upper tube 的picking index有bug 
待优化：人物动作 loading的动画 人物行走方向 并行训练
下一步：加入路网图、修改cutting machine task的选择过程

idea: two-stage RL network

场景的泛化性要考虑： 改变worker 和 machine的数量

# 9.15 
action_space obs_space默认值 等都在rl_task_v1.py这里修改

# 9.16
env reset 不需要再algorithm里面进行 只需要在env、task里面做reset

# 9.19
replay buffer 需要进行修改
rainbow.py 517行 需要改动

如何缩短epoch的时间

# 9.24
设计observation space
# 9.25
task reset
max_episode_length: 10240 need to be change

# 9.29 10.6
输入参数归一化
batch_size暂定为8
history length 设为1
multi step
如何缩短epoch的时间



# 10.8 如何提高训练效率？

vectorized environment。 代码的修改量很大，需要的时间较多
缩短simulation time，等比缩短训练时间。在evaluate的阶段再回到正常仿真时间
采用imitation learning 结合的方法缩短训练时间 

算法上进行简化，去掉distribution RL
没有用到multi-step和这个history length

避免world step


# 10.10
loss曲线为什么会是阶梯状？可能是因为target net 和 online net，也可能是nosie net

# 10.16
1. 进一步缩短训练时间
2. 可视化训练效果，找到为什么reward 会低一些的原因

# 10.17
rule-based:
['hoop_preparing', 'bending_tube_preparing', 'hoop_loading_inner', 'bending_tube_loading_inner', 'collect_product', 'cutting_cube', 'hoop_loading_outer', 'bending_tube_loading_outer', 'cutting_cube', 'hoop_loading_inner', 'bending_tube_loading_inner', 'cutting_cube', 'cutting_cube', 'hoop_loading_inner', 'bending_tube_loading_inner', 'hoop_preparing', 'bending_tube_preparing', 'placing_product', 'hoop_loading_inner', 'bending_tube_loading_inner', 'collect_product', 'cutting_cube', 'placing_product']
length = 23

add two actions
###### 11.21 
如果还是不行的话 就看看别人的loss function怎么做的，以及加上efficient buffer
# 10.21
如何调整weld station的inner还是outer的选择方式？
原来的是cube in list 之后然后再做选择


self.proc_groups_inner_list 要改
self.proc_groups_outer_list 也要改

改hoop loading 以及bending tube loading的规则

placing cube时 选择inner还是outer的规则也要优化，优先选已经完成hoop loading的station
# 10.22 提速训练

修改worker agv运输的时长 done
增加welding 的时间 done
调整回正常训练：两个地方要改回去
加快除锈和抓手时间 done

# 10.23
！！collect product 以及loading的环节 的规则比较难写 可以尝试用RL做突破 这个地方比较灵活！
957 inner length 870 outer length 
max env step 也要改
三段式exploration

# 10.24
可以检查一下random exploration的指标情况 有助于改善reward 函数

# 10.25
验证训练结果
11852  966 len
['bending_tube_preparing', 'hoop_preparing', 'collect_product', 'cutting_cube', 'bending_tube_loading_outer', 'hoop_loading_inner', 'bending_tube_loading_inner', 'hoop_loading_outer', 'cutting_cube', 'bending_tube_loading_inner', 'cutting_cube', 'hoop_loading_inner', 'bending_tube_loading_outer', 'bending_tube_preparing', 'hoop_loading_outer', 'cutting_cube', 'hoop_preparing', 'bending_tube_loading_inner', 'hoop_loading_inner', 'cutting_cube', 'placing_product', 'collect_product', 'placing_product']

['bending_tube_preparing', 'hoop_preparing', 'collect_product', 'cutting_cube', 'bending_tube_loading_outer', 'hoop_loading_inner', 'bending_tube_loading_inner', 'hoop_loading_outer', 'cutting_cube', 'bending_tube_loading_inner', 'cutting_cube', 'hoop_loading_inner', 'bending_tube_loading_outer', 'bending_tube_preparing', 'hoop_loading_outer', 'cutting_cube', 'hoop_preparing', 'bending_tube_loading_inner', 'hoop_loading_inner', 'cutting_cube', 'placing_product', 'collect_product', 'placing_product']

12919
981
['bending_tube_preparing', 'hoop_preparing', 'collect_product', 'cutting_cube', 'bending_tube_loading_outer', 'hoop_loading_inner', 'bending_tube_loading_inner', 'hoop_loading_outer', 'cutting_cube', 'bending_tube_loading_inner', 'cutting_cube', 'hoop_loading_inner', 'bending_tube_loading_outer', 'bending_tube_preparing', 'hoop_loading_outer', 'cutting_cube', 'hoop_preparing', 'bending_tube_loading_inner', 'hoop_loading_inner', 'cutting_cube', 'placing_product', 'collect_product', 'placing_product']


14079 981
 ['bending_tube_preparing', 'hoop_preparing', 'collect_product', 'cutting_cube', 'bending_tube_loading_outer', 'hoop_loading_outer', 'bending_tube_loading_inner', 'hoop_loading_inner', 'cutting_cube', 'bending_tube_loading_inner', 'cutting_cube', 'hoop_loading_inner', 'bending_tube_loading_outer', 'bending_tube_preparing', 'hoop_loading_outer', 'cutting_cube', 'hoop_preparing', 'bending_tube_loading_inner', 'hoop_loading_inner', 'cutting_cube', 'placing_product', 'collect_product', 'placing_product']
# 10.27
length 加 assert
save model的方式要改一下 可以通过evaluate 的方式


### bug meet 
Failed to create change watch
https://forums.developer.nvidia.com/t/isaac-sim-2022-2-is-no-longer-running/239245

https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html#q11-getting-spam-of-failures-failed-to-create-change-watch-for-xxx-errno-28-no-space-left-on-device

### rainbowmini 的loss很大
纠正：1.改用softmax
2.改用mse
3.加clip

idea: buffer
demonstrarion: frequency

##### 11.28 mini 
改了网络输出和结构
mse loss 好像有问题
后续可以吧route全部存在cuda里面加速

### 需要变换实验setting，甚至修改state 信息得包含worker agv的信息

ablation study 可以加一个history state
log 显示上可以加一个进度条

#### 11.28
对于worker agv的状态表示不够， 以及可以通过route map增加随机性
终端log、tansformer、数量变化（env_length这个setting可能也需要变化）

## 12.1
把tokens 加上mask
## 12.2
facing bugs, environment simulation

## 12.3
num worker:1, num agv&box:1, env_length:1132
num worker:1, num agv&box:1, env_length:1012
num worker:1, num agv&box:1, env_length:1029
num worker:1, num agv&box:1, env_length:1034

num worker:1, num agv&box:2, env_length:1149
num worker:1, num agv&box:2, env_length:1110
num worker:1, num agv&box:2, env_length:1148
num worker:1, num agv&box:2, env_length:1129

num worker:1, num agv&box:3, env_length:1148
num worker:1, num agv&box:3, env_length:1138
num worker:1, num agv&box:3, env_length:1148
num worker:1, num agv&box:3, env_length:1127

num worker:2, num agv&box:1, env_length:756
num worker:2, num agv&box:1, env_length:767
num worker:2, num agv&box:1, env_length:751

num worker:2, num agv&box:2, env_length:789
num worker:2, num agv&box:2, env_length:799
num worker:2, num agv&box:2, env_length:800
num worker:2, num agv&box:2, env_length:795

num worker:2, num agv&box:3, env_length:909
num worker:2, num agv&box:3, env_length:900

num worker:3, num agv&box:1, env_length:752
num worker:3, num agv&box:1, env_length:767
num worker:3, num agv&box:1, env_length:750
num worker:3, num agv&box:1, env_length:758

num worker:3, num agv&box:2, env_length:843
num worker:3, num agv&box:2, env_length:695
num worker:3, num agv&box:2, env_length:855
num worker:3, num agv&box:2, env_length:699

num worker:3, num agv&box:3, env_length:872
num worker:3, num agv&box:3, env_length:871
num worker:3, num agv&box:3, env_length:709
num worker:3, num agv&box:3, env_length:838
num worker:3, num agv&box:3, env_length:677
num worker:3, num agv&box:3, env_length:673
num worker:3, num agv&box:3, env_length:694
num worker:3, num agv&box:3, env_length:1024
num worker:3, num agv&box:3, env_length:852

rainbow 的action = None 要改，Evaluate epoch 需要取消注释

# evaluate 环节要做的
test instance 100次 保证可以复现

makespan performance
training performance
甘特图单个例子
zero shot

ablation study
甘特图例子
makespan
zero shot, 记得要改material的相关函数和变量

max_length 也要改

# 注
每一次修改关于observation的变量
1. env的task length 
2. model里的dim_state和网络结构
3. memory要改
4. reset_update函数要改
5. 死循环很可能是memory的问题



## 实验：
Efficient greedy rainbow: 4070 


Efficient nosiy rainbow: FactoryTaskAllocationMiC_2024-12-08_15-44-10


Efficient greedy+noisy rainbow: 
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=FactoryTaskAllocationMiCepsilon_noisy headless=True wandb_activate=True test=True load_dir=/FactoryTaskAllocationMiC_2024-12-09_14-31-02/nn load_name=/FactoryTaskAllocationMiC_ep_25000.pth wandb_project=test_HRTA test_times=100

Edqn(greedy): 
    python omniisaacgymenvs/scripts/rlgames_train_v1.py task=FactoryTaskAllocationMiC train=edqn headless=True wandb_activate=True test=True load_dir=/FactoryTaskAllocationMiC_2024-12-11_13-29-49/nn load_name=/FactoryTaskAllocationMiC_ep_25000.pth wandb_project=test_HRTA test_times=100
    
EDDQN(no dueling):
DQNX(no efficient)


zero shot:
Edqn(greedy): test_edqn_ep_10500.pth_2024-12-11_13-29-49

Efficient greedy rainbow: 4070 test_rainbowepsilon_ep_19500.pth_2024-12-08_17-36-58 全对

Efficient greedy+noisy rainbow: test_epsilon_noisy_ep_14300.pth_2024-12-09_14-31-02

best: test_FactoryTaskAllocationMiC_2024-12-07_14-12-05 测试模型结果："/FactoryTaskAllocationMiC_ep_11100.pth" 的performance最优（"/FactoryTaskAllocationMiC_2024-12-06_17-13-47/nn"
1. 包括training 的曲线, evaluation的：efficient buffer，dueling net, nosiy net，prioritied
2. 对比方法：原版, 没有efficient buffer，dueling net, nosiy net，prioritied (
    EDDQNX-GN (with noisy and greedy) , EDDQNX-G (only greedy), EDDQNX-N (only nosiy), EDQN-G (no dueling, but greedy), DQN)
   1. 变换num_worker, num_robot后对应方法的mean makespan，固定materials
   2. 固定num_worker, num_robot,变换对应的materials, 统计mean makespan (zero_shot)
   3. table统计 make span和 task failed
3. 甘特图分析
4. low_level agent 路径规划
5. 行为分析
6. 超参数





# ignore
__pycache___
__pycache__/
**/runs/
omniisaacgymenvs.egg-info/
checkpoints
omniisaacgymenvs/wandb
**/__pycache__
wandb/


json:
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "test:rainbowmini",
            "type": "debugpy",
            "request": "launch",
            "python": "/home/xue/Repos/miniconda3/envs/isaac-sim/bin/python",
            // "python": "/home/xue/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/bin/python3",
            "program": "${workspaceFolder}/omniisaacgymenvs/scripts/rlgames_train_v1.py",
            "console": "integratedTerminal",
            "stopOnEntry": false,
            // "args": ["-eph", "./pretrain_info/embed_mat.txt", "-mn", "simple_hgn", "-sp", "True", "-debug", "True"],
            "args": ["task=FactoryTaskAllocationMiC", "train=FactoryTaskAllocationMiCRainbowmini", "headless=True", "wandb_activate=True", "test=True"],
            // "args": ["task=FactoryTaskAllocationMiC", "train=FactoryTaskAllocationMiCRainbowmini"],
            "cwd": "${workspaceFolder}",
            "preLaunchTask": "setup conda env",
            // "justMyCode":false  
            
        },
    ]
}


            {
                "name": "test:rainbowmini",
                "type": "debugpy",
                "request": "launch",
                // "program": "${workspaceFolder}/install/map_visualize/lib/map_visualize/get_path",
                "program": "${workspaceFolder}/omniisaacgymenvs/scripts/rlgames_train_v1.py",
                "console": "integratedTerminal",
                "stopOnEntry": false,
                "args": ["task=FactoryTaskAllocationMiC", "train=FactoryTaskAllocationMiCRainbowmini", "wandb_activate=True", "headless=True", "test=True", "rule_based=True"],
                "cwd": "${workspaceFolder}",
                
            },   
