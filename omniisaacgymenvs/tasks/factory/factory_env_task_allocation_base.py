# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: class for nut-bolt env.

Inherits base class and abstract environment class. Inherited by nut-bolt task classes. Not directly executed.

Configuration defined in FactoryEnvNutBolt.yaml. Asset info defined in factory_asset_info_obj.yaml.
"""


import hydra
import numpy as np
import torch

from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.base.rl_task_v1 import RLTask
from omni.physx.scripts import physicsUtils, utils

from omniisaacgymenvs.tasks.factory.factory_base import FactoryBase
from omniisaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from omniisaacgymenvs.tasks.factory.factory_schema_config_env import (
    FactorySchemaConfigEnv,
)

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.articulations import ArticulationView

from omni.usd import get_world_transform_matrix, get_local_transform_matrix
from omniisaacgymenvs.utils.geometry import quaternion  
from omniisaacgymenvs.utils.hybridAstar import hybridAStar 
from omni.isaac.core.utils.prims import set_prim_visibility
import matplotlib.pyplot as plt
from omniisaacgymenvs.utils.hybridAstar import hybridAStar 
import pickle, os

def world_pose_to_navigation_pose(world_pose):
    position, orientation = world_pose[0][0].cpu().numpy(), world_pose[1][0].cpu().numpy()
    euler_angles = quaternion.quaternionToEulerAngles(orientation)
    nav_pose = [position[0], position[1], euler_angles[2]]
    return nav_pose

class Materials(object):

    def __init__(self, cube_list : list, hoop_list : list, bending_tube_list : list, upper_tube_list: list, product_list : list) -> None:

        self.cube_list = cube_list
        self.upper_tube_list = upper_tube_list
        self.hoop_list = hoop_list
        self.bending_tube_list = bending_tube_list
        self.product_list = product_list

        self.cube_state_dic = {-1:"done", 0:"wait", 1:"in_list", 2:"conveying", 3:"conveyed", 4:"cutting", 5:"cut_done", 6:"pick_up_place_cut", 
                                   7:"placed_station_inner", 8:"placed_station_outer", 9:"welding_left", 10:"welding_right", 11:"welding_upper",
                                   12:"process_done", 13:"pick_up_place_product"}
        self.hoop_state_dic = {-1:"done", 0:"wait", 1:"in_box", 2:"on_table", 3:"in_list", 4:"loading", 5:"loaded"}
        self.bending_tube_state_dic = {-1:"done", 0:"wait", 1:"in_box",  2:"on_table", 3:"in_list", 4:"loading", 5:"loaded"}
        self.upper_tube_state_dic = {}
        self.product_state_dic = {0:"waitng", 1:'collected', 2:"placed", -1:"finished"}

        self.cube_states = [0]*len(self.cube_list)
        self.hoop_states = [0]*len(self.hoop_list)
        self.bending_tube_states = [0]*len(self.bending_tube_list)
        self.upper_tube_states = [0]*len(self.upper_tube_list)
        self.product_states = [0]*len(self.product_list)
        '''#for workers and agv to conveying the materials'''
        # self.cube_convey_states = [0]*len(self.cube_list)
        self.hoop_state_dic = {0:"wait", 1:"in_box", 2:"on_table"}
        self.bending_tube_state_dic = {0:"wait", 1:"in_box", 2:"on_table"}
        self.hoop_convey_states = [0]*len(self.hoop_list)
        self.bending_tube_convey_states = [0]*len(self.bending_tube_list)
        # self.upper_tube_convey_states = [0]*len(self.upper_tube_list)
        #for belt conveyor
        self.cube_convey_index = -1
        #cutting machine
        self.cube_cut_index = -1
        #grippers
        self.pick_up_place_cube_index = -1
        self.pick_up_place_upper_tube_index = -1
        #for inner station
        self.inner_hoop_processing_index = -1
        self.inner_cube_processing_index = -1
        self.inner_bending_tube_processing_index = -1
        self.inner_upper_tube_processing_index = -1
        #for outer station
        self.outer_hoop_processing_index = -1
        self.outer_cube_processing_index = -1   #equal to product processing index
        self.outer_bending_tube_processing_index = -1
        self.outer_upper_tube_processing_index = -1

        self.initial_hoop_pose = None
        self.initial_bending_tube_pose = None


        position = [[[-14.44042, 4.77828, 0.6]], [[-13.78823, 4.77828, 0.6]], [[-14.44042, 5.59765, 0.6]], [[-13.78823, 5.59765, 0.6]]]
        orientation = [[1 ,0 ,0, 0]]
        self.position_depot_hoop, self.orientation_depot_hoop = torch.tensor(position, dtype=torch.float32), torch.tensor(orientation, dtype=torch.float32)
        
        position = [[[-31.64901, 4.40483, 1.1]], [[-30.80189, 4.40483, 1.1]], [[-31.64901, 5.31513, 1.1]], [[-30.80189, 5.31513, 1.1]]]
        orientation = [[-1.6081e-16, -6.1232e-17,  1.0000e+00, -6.1232e-17]]
        self.position_depot_bending_tube, self.orientation_depot_bending_tube = torch.tensor(position, dtype=torch.float32), torch.tensor(orientation, dtype=torch.float32)

        position = [[[-35., 15., 0]], [[-35, 16, 0]], [[-35, 17, 0]], [[-35, 18, 0]], [[-35, 19, 0]]]
        orientation = [[1 ,0 ,0, 0]]
        self.position_depot_product, self.orientation_depot_product = torch.tensor(position, dtype=torch.float32), torch.tensor(orientation, dtype=torch.float32)

        in_box_offsets = [[[0.5,0.5, 0.5]], [[-0.5,0.5, 0.5]], [[0.5,-0.5, 0.5]], [[-0.5, -0.5, 0.5]]]
        self.in_box_offsets = torch.tensor(in_box_offsets, dtype=torch.float32) 



    def get_world_poses(self, list):
        poses = []
        for obj in list:
            poses.append(obj.get_world_poses())
        return poses
    
    def update_poses(self):
        pass

    def done(self):
        return min(self.product_states) == 2
    
    def produce_product_req(self):
        try:
            self.product_states.index(0)
            return True
        except: 
            return False

    def find_next_raw_cube_index(self):
        # index 
        try:
            return self.cube_states.index(0)
        except:
            return -1
        # return self.cube_states.index(0)
    
    def find_next_raw_upper_tube_index(self):
        # index 
        try:
            return self.upper_tube_states.index(0)
        except:
            return -1
        # return self.upper_tube_states.index(0)
    
    def find_next_raw_hoop_index(self):
        # index 2 is on table
        try:
            return self.hoop_states.index(2)
        except:
            return -1
        # return self.hoop_states.index(0)
    
    def find_next_raw_bending_tube_index(self):
        # index 
        try:
            return self.bending_tube_states.index(2)
        except:
            return -1


class Characters(object):

    def __init__(self, character_list) -> None:
        self.num = len(character_list)
        self.list = character_list
        self.state_character_dic = {0:"free", 1:"approaching", 2:"waiting_box", 3:"putting_in_box", 4:"putting_on_table"}
        self.task_range = {'hoop_preparing', 'bending_tube_preparing', 'hoop_loading_inner', 'bending_tube_loading_inner', 'hoop_loading_outer', 'bending_tube_loading_outer', "cutting_cube", 
                           'placing_product'}
        self.sub_task_character_dic = {0:"free", 1:"put_hoop_into_box", 2:"put_bending_tube_into_box", 3:"put_hoop_on_table", 4:"put_bending_tube_on_table", 
                                    5:'hoop_loading_inner', 6:'bending_tube_loading_inner', 7:'hoop_loading_outer', 8: 'bending_tube_loading_outer', 9: 'cutting_cube', 10:'placing_product'}
        
        self.low2high_level_task_dic = {"put_hoop_into_box":"hoop_preparing", "put_bending_tube_into_box":'bending_tube_preparing', "put_hoop_on_table":'hoop_preparing', 
                        "put_bending_tube_on_table":'bending_tube_preparing', 'hoop_loading_inner':'hoop_loading_inner', 'bending_tube_loading_inner':'bending_tube_loading_inner', 
                        'hoop_loading_outer':'hoop_loading_outer', 'bending_tube_loading_outer': 'bending_tube_loading_outer', 'cutting_cube': 'cutting_cube', 'placing_product':'placing_product'}
        
        self.poses_dic = {"put_hoop_into_box": [1.28376, 6.48821, np.deg2rad(0)] , "put_bending_tube_into_box": [1.28376, 13.12021, np.deg2rad(0)], 
                        "put_hoop_on_table": [-12.26318, 4.72131, np.deg2rad(0)], "put_bending_tube_on_table":[-32, 8.0, np.deg2rad(-90)],
                        'hoop_loading_inner':[-16.26241, 6.0, np.deg2rad(180)],'bending_tube_loading_inner':[-29.06123, 6.3725, np.deg2rad(0)],
                        'hoop_loading_outer':[-16.26241, 6.0, np.deg2rad(180)], 'bending_tube_loading_outer': [-29.06123, 6.3725, np.deg2rad(0)],
                        'cutting_cube':[-29.83212, -1.54882, np.deg2rad(0)], 'placing_product':[-40.47391, 12.91755, np.deg2rad(0)],
                        'initial_pose_0':[-11.5768, 6.48821, 0.0], 'initial_pose_1':[-30.516169, 7.5748153, 0.0]}
        # for idx, charc in enumerate(self.list):
        #     xy_yaw = world_pose_to_navigation_pose(charc.get_world_poses())
        #     # self.initial_xy_yaw.append(xy_yaw)
        #     self.poses_dic[f'initial_pose_{idx}'] = xy_yaw
        self.routes_dic = None

        self.states = [0]*self.num
        self.tasks = [0]*self.num
        # self.corresp_agv_idxs = [-1]*self.num
        # self.corresp_box_idxs = [-1]*self.num
        # self.corresp_agvs_idxs = [-1]*self.num
        self.x_paths = [[] for i in range(len(character_list))]
        self.y_paths = [[] for i in range(len(character_list))]
        self.yaws = [[] for i in range(len(character_list))]
        self.path_idxs = [0 for i in range(len(character_list))]

        self.picking_pose_hoop = [1.28376, 6.48821, np.deg2rad(0)] 
        self.picking_pose_bending_tube = [1.28376, 13.12021, np.deg2rad(0)] 
        self.picking_pose_table_hoop = [-12.26318, 4.72131, np.deg2rad(0)]
        self.picking_pose_table_bending_tube = [-32, 8.0, np.deg2rad(-90)]

        self.loading_pose_hoop = [-16.26241, 6.0, np.deg2rad(180)]
        self.loading_pose_bending_tube = [-29.06123, 6.3725, np.deg2rad(0)]

        self.cutting_cube_pose = [-29.83212, -1.54882, np.deg2rad(0)]

        self.placing_product_pose = [-40.47391, 12.91755, np.deg2rad(0)]
        self.PUTTING_TIME = 5
        self.LOADING_TIME = 5
        self.loading_operation_time_steps = [0 for i in range(len(character_list))]
        return
    
    def reset(self, idx):
        if idx < 0 :
            return
        self.states[idx] = 0
        self.tasks[idx] = 0

    def assign_task(self, high_level_task):
        #todo 
        if high_level_task not in self.task_range:
            return -2
        idx = self.find_available_charac()
        if idx == -1:
            return idx
        if high_level_task == 'hoop_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 1 
        elif high_level_task == 'bending_tube_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 2
        elif high_level_task == 'hoop_loading_inner':
            self.tasks[idx] = 5
        elif high_level_task == 'bending_tube_loading_inner':
            self.tasks[idx] = 6
        elif high_level_task == 'hoop_loading_outer':
            self.tasks[idx] = 7
        elif high_level_task == 'bending_tube_loading_outer':
            self.tasks[idx] = 8
        elif high_level_task == 'cutting_cube':
            #todo
            for _idx in range(0, len(self.list)):
                xyz, _ = self.list[_idx].get_world_poses()
                idx = -1
                if self.tasks[_idx] == 0 and xyz[0][0] < -22:
                    self.tasks[_idx] = 9
                    idx = _idx
                    return idx
            # if self.tasks[1] == 0: #only assign worker 1 to do the cutting cube task 
            #     self.tasks[1] = 9
            #     idx = 1
            # else:
            #     return -1
        elif high_level_task == 'placing_product':
            self.tasks[idx] = 10
        return idx
    
    def find_available_charac(self, idx=0):
        try:
            return self.tasks.index(idx)
        except: 
            return -1

    def step_next_pose(self, charac_idx = 0):
        reaching_flag = False
        #skip the initial pose
        # if len(self.x_paths[agv_idx]) == 0:
        #     position = [current_pose[0], current_pose[1], 0]
        #     euler_angles = [0,0, current_pose[2]]
        #     return position, quaternion.eulerAnglesToQuaternion(euler_angles), True

        self.path_idxs[charac_idx] += 1
        path_idx = self.path_idxs[charac_idx]
        # if agv_idx == 0:
        #     a = 1
        if path_idx == (len(self.x_paths[charac_idx]) - 1):
            reaching_flag = True
            position = [self.x_paths[charac_idx][-1], self.y_paths[charac_idx][-1], 0]
            euler_angles = [0,0, self.yaws[charac_idx][-1]]
        else:
            position = [self.x_paths[charac_idx][path_idx], self.y_paths[charac_idx][path_idx], 0]
            euler_angles = [0,0, self.yaws[charac_idx][path_idx]]

        orientation = quaternion.eulerAnglesToQuaternion(euler_angles)
        return position, orientation, reaching_flag

    
    def reset_path(self, charac_idx):
        self.x_paths[charac_idx] = []
        self.y_paths[charac_idx] = []
        self.yaws[charac_idx] = []
        self.path_idxs[charac_idx] = 0

    def low2high_level_task_mapping(self, task):
        task = self.sub_task_character_dic[task]
        if task in self.low2high_level_task_dic.keys():
            return self.low2high_level_task_dic[task]
        else: return -1


class Agvs(object):

    def __init__(self, agv_list) -> None:
        self.list = agv_list
        self.num = len(agv_list)
        self.state_dic = {0:"free", 1:"moving_to_box", 2:"carrying_box", 3:"waiting"}
        self.sub_task_dic = {0:"free", 1:"carry_box_to_hoop", 2:"carry_box_to_bending_tube", 3:"carry_box_to_hoop_table", 4:"carry_box_to_bending_tube_table", 5:'collect_product', 6:'placing_product'}
        self.task_range = {'hoop_preparing', 'bending_tube_preparing', 'collect_product','placing_product'}
        self.low2high_level_task_dic =  {"carry_box_to_hoop":'hoop_preparing', "carry_box_to_bending_tube":'bending_tube_preparing', "carry_box_to_hoop_table":'hoop_preparing', 
                                         "carry_box_to_bending_tube_table":'bending_tube_preparing', 'collect_product':'collect_product', 'placing_product':'placing_product'}
        
        self.poses_dic = {"carry_box_to_hoop": [-0.654, 8.0171, np.deg2rad(0)] , "carry_box_to_bending_tube": [-0.654, 11.62488, np.deg2rad(0)], 
                        "carry_box_to_hoop_table": [-11.69736, 5.71486, np.deg2rad(0)], "carry_box_to_bending_tube_table":[-33.55065, 5.71486, np.deg2rad(-90)] ,
                        'collect_product':[-21.76757, 10.78427, np.deg2rad(0)],'placing_product':[-38.54638, 12.40097, np.deg2rad(0)], 
                        'initial_pose_0':[-4.8783107, 8.017096, 0.0], 'initial_pose_1': [-4.8726454, 11.656976, 0.0],
                        'initial_box_pose_0': [-1.6895515, 8.0171, 0.0], 'initial_box_pose_1': [-1.7894887, 11.822739, 0.0]}
        
        # self.initial_xy_yaw = []
        # for idx, agv in enumerate(self.list):
        #     xy_yaw = world_pose_to_navigation_pose(agv.get_world_poses())
        #     # self.initial_xy_yaw.append(xy_yaw)
        #     self.poses_dic[f'initial_pose_{idx}'] = xy_yaw
        self.routes_dic = None

        self.states = [0]*self.num
        self.tasks = [0]*self.num
        # self.corresp_charac_idxs = [-1]*self.num
        # self.corresp_box_idxs = [-1]*self.num

        self.x_paths = [[] for i in range(len(agv_list))]
        self.y_paths = [[] for i in range(len(agv_list))]
        self.yaws = [[] for i in range(len(agv_list))]
        self.path_idxs = [0 for i in range(len(agv_list))]

        self.picking_pose_hoop = [-0.654, 8.0171, np.deg2rad(0)]  #
        self.picking_pose_bending_tube = [-0.654, 11.62488, np.deg2rad(0)]
        self.picking_pose_table_hoop = [-11.69736, 5.71486, np.deg2rad(0)]
        self.picking_pose_table_bending_tube = [-33.55065, 5.71486, np.deg2rad(-90)] 
        self.collecting_product_pose = [-21.76757, 10.78427, np.deg2rad(0)]
        self.placing_product_pose = [-38.54638, 12.40097, np.deg2rad(0)]
        return
    
    def reset(self, idx):
        if idx < 0 :
            return
        self.tasks[idx] = 0
        self.states[idx] = 0

    def assign_task(self, high_level_task, box_idx, box_xyz):
        #todo  
        if high_level_task not in self.task_range:
            return -2
        idx = self.find_available_agv(box_idx, box_xyz)
        if idx == -1:
            return idx
        if high_level_task == 'hoop_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 1 
        elif high_level_task == 'bending_tube_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 2
        elif high_level_task == 'collect_product':
            self.tasks[idx] = 5
        elif high_level_task == 'placing_product':
            self.tasks[idx] = 6 
        return idx
    
    def find_available_agv(self, box_idx, box_xyz):
        if box_idx == -1: 
            try:
                return self.tasks.index(0)
            except: 
                return -1
        else:
            min_dis_idx = -1
            pre_dis = torch.inf
            for agv_idx in range(0, len(self.list)):
                if self.tasks[agv_idx] == 0:
                    agv_xyz, _ = self.list[agv_idx].get_world_poses()
                    dis = torch.norm(agv_xyz[0] - box_xyz)
                    if dis.cpu() < pre_dis:
                        pre_dis = dis
                        min_dis_idx = agv_idx
            return min_dis_idx
    
    def step_next_pose(self, agv_idx):
        reaching_flag = False
        #skip the initial pose
        # if len(self.x_paths[agv_idx]) == 0:
        #     position = [current_pose[0], current_pose[1], 0]
        #     euler_angles = [0,0, current_pose[2]]
        #     return position, quaternion.eulerAnglesToQuaternion(euler_angles), True

        self.path_idxs[agv_idx] += 1
        path_idx = self.path_idxs[agv_idx]
        # if agv_idx == 0:
        #     a = 1
        if path_idx >= (len(self.x_paths[agv_idx])):
            reaching_flag = True
            position = [self.x_paths[agv_idx][-1], self.y_paths[agv_idx][-1], 0]
            euler_angles = [0,0, self.yaws[agv_idx][-1]]
        else:
            position = [self.x_paths[agv_idx][path_idx], self.y_paths[agv_idx][path_idx], 0]
            euler_angles = [0,0, self.yaws[agv_idx][path_idx]]

        orientation = quaternion.eulerAnglesToQuaternion(euler_angles)
        return position, orientation, reaching_flag
    
    def reset_path(self, agv_idx):
        self.x_paths[agv_idx] = []
        self.y_paths[agv_idx] = []
        self.yaws[agv_idx] = []
        self.path_idxs[agv_idx] = 0
    
    def low2high_level_task_mapping(self, task):
        task = self.sub_task_dic[task]
        if task in self.low2high_level_task_dic.keys():
            return self.low2high_level_task_dic[task]
        else: return -1



class TransBoxs(object):

    def __init__(self, box_list) -> None:
        self.list = box_list
        self.num = len(box_list)
        self.state_dic = {0:"free", 1:"waiting", 2:"moving"}
        self.sub_task_dic = {0:"free", 1:"waiting_agv", 2:"moving_with_box", 3: "collect_product"}
        self.task_range = {'hoop_preparing', 'bending_tube_preparing', 'collect_product','placing_product'}

        self.poses_dic = {'initial_pose_0': [-1.6895515, 8.0171, 0.0], 'initial_pose_1': [-1.7894887, 11.822739, 0.0]}

        # for idx, box in enumerate(self.list):
        #     xy_yaw = world_pose_to_navigation_pose(box.get_world_poses())
        #     # self.initial_xy_yaw.append(xy_yaw)
        #     self.poses_dic[f'initial_pose_{idx}'] = xy_yaw
        self.states = [0]*self.num
        self.tasks = [0]*self.num
        self.high_level_tasks = ['']*self.num
        # self.corresp_agv_idxs = [-1]*self.num
        # self.corresp_charac_idxs = [-1]*self.num

        # self.picking_pose_hoop = [-0.09067, 6.48821, np.deg2rad(180)]
        # self.picking_pose_bending_tube = [-0.09067, 13.12021, np.deg2rad(0)]

        self.hoop_idx_list =[[] for i in range(len(box_list))]
        self.bending_tube_idx_sets = [set() for i in range(len(box_list))]
        self.product_idx_list = [[] for i in range(len(box_list))]
        self.CAPACITY = 4
        self.counts = [0 for i in range(len(box_list))]

        self.product_collecting_idx = -1

        return
    
    def reset(self, idx):
        if idx < 0 :
            return
        self.tasks[idx] = 0
        self.states[idx] = 0
        self.high_level_tasks[idx] = ''
        # self.corresp_charac_idxs[idx] = -1
        # self.corresp_agv_idxs[idx] = -1

    def assign_task(self, high_level_task):
        #todo
        if high_level_task not in self.task_range:
            return -2
        
        if high_level_task == 'placing_product':
            idx = self.find_carrying_products_box_idx()
            if idx >=0 :
                self.high_level_tasks[idx] = high_level_task
                self.tasks[idx] = 1 
                return idx
            else:
                return -1 
        
        idx = self.find_available_box()
        if idx == -1:
            if high_level_task == 'collect_product':
                self.product_collecting_idx = -1
            return -1
        # if high_level_task == 'hoop_preparing' or high_level_task == 'bending_tube_preparing' or high_level_task == 'colle':
            # idx = self.find_available_charac()
        else:
            self.high_level_tasks[idx] = high_level_task
            self.tasks[idx] = 1 
            if high_level_task == 'collect_product':
                self.product_collecting_idx = idx
            return idx

    def find_available_box(self):
        try:
            return self.tasks.index(0)
        except: 
            return -1
        
    def find_carrying_products_box_idx(self):
        for list, idx in zip(self.product_idx_list, range(len(self.product_idx_list))):
            if len(list) > 0:
                return idx
        else:
            return -1
        
    def find_full_products_box_idx(self):
        for list, idx in zip(self.product_idx_list, range(len(self.product_idx_list))):
            if len(list) >= self.CAPACITY:
                return idx
        else:
            return -1
    
        
    def is_full_products(self):
        return self.find_full_products_box_idx() != -1
    


class TaskManager(object):
    def __init__(self, character_list, agv_list, box_list) -> None:
        self.characters = Characters(character_list=character_list)
        self.agvs = Agvs(agv_list = agv_list)
        self.boxs = TransBoxs(box_list=box_list)
        self.task_dic =  {0: 'hoop_preparing', 1:'bending_tube_preparing', 2:'hoop_loading_inner', 3:'bending_tube_loading_inner', 4:'hoop_loading_outer', 
                          5:'bending_tube_loading_outer', 6:'cutting_cube', 7:'collect_product', 8:'placing_product'}
        self.task_in_set = set()
        self.task_in_dic = {}
        return
    
    def assign_task(self, task):
        
        charac_idx = self.characters.assign_task(task)
        box_idx = self.boxs.assign_task(task)
        box_xyz, _ = self.boxs.list[box_idx].get_world_poses()
        agv_idx = self.agvs.assign_task(task, box_idx, box_xyz[0])
        
        lacking_resource = False
        if charac_idx == -1 or agv_idx == -1 or box_idx == -1:
            lacking_resource = True            

        self.task_in_set.add(task)
        self.task_in_dic[task] = {'charac_idx': charac_idx, 'agv_idx': agv_idx, 'box_idx': box_idx, 'lacking_resource': lacking_resource}

        return True

    def task_clearing(self, task):

        charac_idx, agv_idx, box_idx = self.task_in_dic[task]['charac_idx'], self.task_in_dic[task]['agv_idx'], self.task_in_dic[task]['box_idx']
        self.characters.reset(charac_idx)
        self.agvs.reset(agv_idx)
        self.boxs.reset(box_idx)
        self.task_in_set.remove(task)
        del self.task_in_dic[task]

        return

    def step(self):
        for task in self.task_in_set:
            if self.task_in_dic[task]['lacking_resource']:
                if self.task_in_dic[task]['charac_idx'] == -1:
                    self.task_in_dic[task]['charac_idx'] = self.characters.assign_task(task)
                if self.task_in_dic[task]['box_idx'] == -1:
                    self.task_in_dic[task]['box_idx'] = self.boxs.assign_task(task)
                    #reset agv idx and find the suitable agv for box again
                    agv_idx = self.task_in_dic[task]['agv_idx']
                    if agv_idx >= 0:
                        self.agvs.reset(agv_idx)
                        self.task_in_dic[task]['agv_idx'] = -1
                if self.task_in_dic[task]['agv_idx'] == -1:
                    box_idx = self.task_in_dic[task]['box_idx']
                    box_xyz, _ = self.boxs.list[box_idx].get_world_poses()
                    self.task_in_dic[task]['agv_idx'] = self.agvs.assign_task(task, box_idx, box_xyz)

                try:
                    list(self.task_in_dic[task].values()).index(-1)
                    self.task_in_dic[task]['lacking_resource'] = True
                except: 
                    self.task_in_dic[task]['lacking_resource'] = False
                
        return 

    def corresp_charac_agv_box_idx(self, task):
        if task not in self.task_in_dic.keys():
            return -1, -1, -1
        return self.task_in_dic[task]['charac_idx'], self.task_in_dic[task]['agv_idx'], self.task_in_dic[task]['box_idx']
    

class FactoryEnvTaskAlloc(FactoryBase, FactoryABCEnv):
    def __init__(self, name, sim_config, env) -> None:
        """Initialize base superclass. Initialize instance variables."""

        super().__init__(name, sim_config, env)

        self._get_env_yaml_params()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)

        config_path = (
            "task/FactoryEnvTaskAllocation.yaml"  # relative to Hydra search path (cfg dir)
        )
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting

        asset_info_path = "../tasks/factory/yaml/factory_asset_info_task_allocation.yaml"
        self.asset_info_obj = hydra.compose(config_name=asset_info_path)
        self.asset_info_obj = self.asset_info_obj[""][""][""]["tasks"][
            "factory"
        ][
            "yaml"
        ]  # strip superfluous nesting

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_observations = self._task_cfg["env"]["numObservations"]
        self._num_actions = self._task_cfg["env"]["numActions"]
        self._env_spacing = self.cfg_base["env"]["env_spacing"]

        self._get_env_yaml_params()

    def set_up_scene(self, scene) -> None:
        """Import assets. Add to scene."""
        # Increase buffer size to prevent overflow for Place and Screw tasks
        physxSceneAPI = self.world.get_physics_context()._physx_scene_api
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(256 * 1024 * 1024)

        # self.import_franka_assets(add_to_stage=True)
        # /obj/multi_people/F_Business_02/female_adult_business_02
        
        # from pxr import Sdf
        # # prim_path = Sdf.Path(f"/World/envs/env_0" + "/obj/multi_people/F_Business_02/female_adult_business_02/ManRoot/male_adult_construction_05")
        # prim_path = Sdf.Path(f"/World/envs/env_0" + "/obj/Characters/male_adult_construction_05/ManRoot/male_adult_construction_05")

        # # /obj/Characters/male_adult_construction_05/ManRoot/male_adult_construction_05
        # # from omniisaacgymenvs.robots.omni_anim_people.scripts.character_behavior import CharacterBehavior
        # # self.character_0 = CharacterBehavior(prim_path)
        # import omni.anim.graph.core as ag
        # self.character = ag.get_character(str(prim_path))

        self._stage = get_current_stage()    
        # self.create_nut_bolt_material()
        RLTask.set_up_scene(self, scene, replicate_physics=False)
        self._import_env_assets(add_to_stage=True)

        # self.frankas = FactoryFrankaView(
        #     prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        # )

        #debug
        stage_utils.print_stage_prim_paths()
                
        for i in range(0, self.num_envs):
            ground_prim = self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/GroundPlane")
            set_prim_visibility(prim=ground_prim, visible=False)

        # perspective = self._stage.GetPrimAtPath(f"/OmniverseKit_Persp")
        # ConveyorNode_0.GetAttribute('inputs:velocity').Set(100)
        # ConveyorNode_0.GetAttribute('inputs:animateTexture').Set(True)
        # perspective.GetAttributes()
        # perspective.GetAttribute('xformOp:translate').Set((36.0, 38.6, 16.8))
        # perspective.GetAttribute('xformOp:rotateXYZ').Set((63.8, 0, 141))
        # translate = perspective.GetAttribute('xformOp:translate')
        # result, prim_ConveyorBelt_A09_0_2 = commands.execute(
        #     "CreateConveyorBelt",
        #     prim_name="ConveyorActionGraph",
        #     conveyor_prim=self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/ConveyorBelt_A09_0_2/Belt")
        # )
        self.obj_belt_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/ConveyorBelt_A09_0_0/Belt",
            name="ConveyorBelt_A09_0_0/Belt",
            track_contact_forces=True,
        )
        self.obj_0_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/obj_0_1",
            name="obj_0_1",
            track_contact_forces=True,
        )
        self.obj_belt_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/ConveyorBelt_A09_0_2/Belt",
            name="ConveyorBelt_A09_0_2/Belt",
            track_contact_forces=True,
        )
        # self.obj_belt_2 = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/obj/ConveyorBelt_A08/Rollers",
        #     name="ConveyorBelt_A08/Rollers",
        #     track_contact_forces=True,
        # )
        self.obj_part_10 = ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part10", name="obj_part_10", reset_xform_properties=False
        )
        self.obj_part_7 = ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part7", name="obj_part_7", reset_xform_properties=False
        )
        self.obj_part_7_manipulator = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part7/manipulator2/robotiq_arg2f_base_link", name="obj_part_7_manipulator", reset_xform_properties=False
        )
        self.obj_part_9_manipulator = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part9/manipulator2/robotiq_arg2f_base_link", name="obj_part_9_manipulator", reset_xform_properties=False
        )
        self.obj_11_station_0 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0", name="obj_11_station_0", reset_xform_properties=False
        )
        self.obj_11_station_0_revolution = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/revolution", name="Station0/revolution", reset_xform_properties=False
        )
        self.obj_11_station_1_revolution = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/revolution", name="Station1/revolution", reset_xform_properties=False
        )
        self.obj_11_station_0_middle = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/middle_left", name="Station0/middle_left", reset_xform_properties=False
        )
        self.obj_11_station_1_middle = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/middle_left", name="Station1/middle_left", reset_xform_properties=False
        )
        self.obj_11_station_0_right = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/right", name="Station0/right", reset_xform_properties=False
        )
        self.obj_11_station_1_right = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/right", name="Station1/right", reset_xform_properties=False
        )
        self.obj_11_station_1 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1", name="obj_11_station_1", reset_xform_properties=False
        )
        self.obj_11_welding_0 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Welding0", name="obj_11_welding_0", reset_xform_properties=False
        )
        self.obj_11_welding_1 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Welding1", name="obj_11_welding_1", reset_xform_properties=False
        )
        self.obj_2_loader_0 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part2/Loaders/Loader0", name="obj_2_loader_0", reset_xform_properties=False
        )
        self.obj_2_loader_1 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part2/Loaders/Loader1", name="obj_2_loader_1", reset_xform_properties=False
        )

        self.materials_cube_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/cubes/cube_0",
            name="cube_0",
            track_contact_forces=True,
        )
        self.materials_hoop_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoops/hoop_0",
            name="hoop_0",
            track_contact_forces=True,
        )
        self.materials_bending_tube_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tubes/bending_tube_0",
            name="bending_tube_0",
            track_contact_forces=True,
        )
        self.materials_upper_tube_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tubes/upper_tube_0",
            name="upper_tube_0",
            track_contact_forces=True,
        )
        self.product_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/products/product_0",
            name="product_0",
            track_contact_forces=True,
        )

        self.materials_cube_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/cubes/cube_01",
            name="cube_1",
            track_contact_forces=True,
        )
        self.materials_hoop_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoops/hoop_01",
            name="hoop_1",
            track_contact_forces=True,
        )
        self.materials_bending_tube_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tubes/bending_tube_01",
            name="bending_tube_1",
            track_contact_forces=True,
        )
        self.materials_upper_tube_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tubes/upper_tube_01",
            name="upper_tube_1",
            track_contact_forces=True,
        )
        self.product_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/products/product_01",
            name="product_1",
            track_contact_forces=True,
        )

        self.materials_cube_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/cubes/cube_02",
            name="cube_2",
            track_contact_forces=True,
        )
        self.materials_hoop_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoops/hoop_02",
            name="hoop_2",
            track_contact_forces=True,
        )
        self.materials_bending_tube_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tubes/bending_tube_02",
            name="bending_tube_2",
            track_contact_forces=True,
        )
        self.materials_upper_tube_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tubes/upper_tube_02",
            name="upper_tube_2",
            track_contact_forces=True,
        )
        self.product_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/products/product_02",
            name="product_2",
            track_contact_forces=True,
        )

        self.materials_cube_3 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/cubes/cube_03",
            name="cube_3",
            track_contact_forces=True,
        )
        self.materials_hoop_3 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoops/hoop_03",
            name="hoop_3",
            track_contact_forces=True,
        )
        self.materials_bending_tube_3 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tubes/bending_tube_03",
            name="bending_tube_3",
            track_contact_forces=True,
        )
        self.materials_upper_tube_3 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tubes/upper_tube_03",
            name="upper_tube_3",
            track_contact_forces=True,
        )
        self.product_3 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/products/product_03",
            name="product_3",
            track_contact_forces=True,
        )


        self.materials_cube_4 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/cubes/cube_04",
            name="cube_4",
            track_contact_forces=True,
        )
        self.materials_hoop_4 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoops/hoop_04",
            name="hoop_4",
            track_contact_forces=True,
        )
        self.materials_bending_tube_4 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tubes/bending_tube_04",
            name="bending_tube_4",
            track_contact_forces=True,
        )
        self.materials_upper_tube_4 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tubes/upper_tube_04",
            name="upper_tube_4",
            track_contact_forces=True,
        )
        self.product_4 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/products/product_04",
            name="product_4",
            track_contact_forces=True,
        )
        scene.add(self.obj_11_station_0)
        scene.add(self.obj_11_station_1)
        scene.add(self.obj_11_welding_0)
        scene.add(self.obj_11_welding_1)
        scene.add(self.obj_2_loader_0)
        scene.add(self.obj_2_loader_1)
        scene.add(self.obj_part_9_manipulator)
        scene.add(self.obj_part_10)
        scene.add(self.obj_part_7)
        # self.obj_cube = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/obj/obj_cube",
        #     name="obj_cube",
        #     track_contact_forces=True,
        # )
        scene.add(self.obj_0_1)
        scene.add(self.obj_belt_0)
        scene.add(self.obj_belt_1)

        scene.add(self.materials_cube_0)
        scene.add(self.materials_hoop_0)
        scene.add(self.materials_bending_tube_0)
        scene.add(self.materials_upper_tube_0)
        scene.add(self.product_0)   

        scene.add(self.materials_cube_1)
        scene.add(self.materials_hoop_1)
        scene.add(self.materials_bending_tube_1)
        scene.add(self.materials_upper_tube_1)
        scene.add(self.product_1) 

        scene.add(self.materials_cube_2)
        scene.add(self.materials_hoop_2)
        scene.add(self.materials_bending_tube_2)
        scene.add(self.materials_upper_tube_2)
        scene.add(self.product_2)
    
        scene.add(self.materials_cube_3)
        scene.add(self.materials_hoop_3)
        scene.add(self.materials_bending_tube_3)
        scene.add(self.materials_upper_tube_3)
        scene.add(self.product_3)
      
        scene.add(self.materials_cube_4)
        scene.add(self.materials_hoop_4)
        scene.add(self.materials_bending_tube_4)
        scene.add(self.materials_upper_tube_4)
        scene.add(self.product_4)


        #materials states
        cube_list = [self.materials_cube_0, self.materials_cube_1, self.materials_cube_2, self.materials_cube_3, self.materials_cube_4]
        hoop_list = [self.materials_hoop_0, self.materials_hoop_1, self.materials_hoop_2, self.materials_hoop_3, self.materials_hoop_4]
        bending_tube_list = [self.materials_bending_tube_0, self.materials_bending_tube_1, self.materials_bending_tube_2, self.materials_bending_tube_3, self.materials_bending_tube_4]
        upper_tube_list = [self.materials_upper_tube_0, self.materials_upper_tube_1, self.materials_upper_tube_2, self.materials_upper_tube_3, self.materials_upper_tube_4]
        product_list = [self.product_0, self.product_1, self.product_2, self.product_3, self.product_4]
        self.materials : Materials = Materials(cube_list=cube_list, hoop_list=hoop_list, bending_tube_list=bending_tube_list, upper_tube_list=upper_tube_list, product_list = product_list)
        # self.materials_flag_dic = {-1:"done", 0:"wait", 1:"conveying", 2:"conveyed", 3:"cutting", 4:"cut_done", 5:"pick_up_cut", 
        # 5:"down", 6:"combine_l", 7:"weld_l", 8:"combine_r", 9:"weld_r"}
        # conveyor
        #0 free 1 working
        self.convey_state = 0
        #cutting machine
        #to do 
        self.cutting_state_dic = {0:"free", 1:"work", 2:"reseting"}
        self.cutting_machine_state = 0
        self.c_machine_oper_time = 0
        #gripper
        # self.max_speed_in_out = 0.1
        # self.max_speed_left_right = 0.1
        # self.max_speed_up_down = 0.1
        # self.max_speed_grip = 0.1
        speed = 0.3
        self.operator_gripper = torch.tensor([speed]*10, device='cuda:0')
        self.gripper_inner_task_dic = {0: "reset", 1:"pick_cut", 2:"place_cut_to_inner_station", 3:"place_cut_to_outer_station", 
                                    4:"pick_product_from_inner", 5:"pick_product_from_outer", 6:"place_product_from_inner", 7:"place_product_from_outer"}
        self.gripper_inner_task = 0
        self.gripper_inner_state_dic = {0: "free_empty", 1:"picking", 2:"placing"}
        self.gripper_inner_state = 0

        self.gripper_outer_task_dic = {0: "reset", 1:"pick_upper_tube_for_inner_station", 2:"pick_upper_tube_for_outer_station", 3:"place_upper_tube_to_inner_station", 4:"place_upper_tube_to_outer_station"}
        self.gripper_outer_task = 0
        self.gripper_outer_state_dic = {0: "free_empty", 1:"picking", 2:"placing"}
        self.gripper_outer_state = 0

        #welder 
        # self.max_speed_welder = 0.1
        self.welder_inner_oper_time = 0
        self.welder_outer_oper_time = 0
        self.operator_welder = torch.tensor([0.2], device='cuda:0')
        self.welder_task_dic = {0: "reset", 1:"weld_left", 2:"weld_right", 3:"weld_middle",}
        self.welder_state_dic = {0: "free_empty", 1: "moving_left", 2:"welding_left", 3:"welded_left", 4:"moving_right",
                                 5:"welding_right", 6:"rotate_and_welding", 7:"welded_right", 8:"welding_middle" , 9:"welded_upper"}
        
        self.welder_inner_task = 0
        self.welder_inner_state = 0
        
        self.welder_outer_task = 0
        self.welder_outer_state = 0
        
        #station
        # self.welder_inner_oper_time = 10
        self.operator_station = torch.tensor([0.1, 0.1, 0.1, 0.1], device='cuda:0')
        self.station_task_left_dic = {0: "reset", 1:"weld"}
        self.station_state_left_dic = {0: "reset_empty", 1:"loading", 2:"rotating", 3:"waiting", 4:"welding", 5:"welded", 6:"finished", -1:"resetting"}
        self.station_task_inner_left = 0
        self.station_task_outer_left = 0
        self.station_state_inner_left = -1
        self.station_state_outer_left = -1

        self.station_middle_task_dic = {0: "reset", 1:"weld_left", 2:"weld_middle", 3:"weld_right"}
        self.station_state_middle_dic = {-1:"resetting", 0: "reset_empty", 1:"placing", 2:"placed", 3:"moving_left", 4:"welding_left", 
                                         5:"welded_left", 6:"welding_right", 7:"welded_right", 8:"welding_upper", 9:"welded_upper"}
        self.station_state_inner_middle = 0
        self.station_state_outer_middle = 0
        self.station_task_inner_middle = 0
        self.station_task_outer_middle = 0
        
        self.station_right_task_dic = {0: "reset", 1:"weld"}
        self.station_state_right_dic = {0: "reset_empty", 1:"placing", 2:"placed", 3:"moving", 4:"welding_right", -1:"resetting"}
        self.station_state_inner_right = 0
        self.station_state_outer_right = 0
        self.station_task_outer_right = 0
        self.station_task_inner_right = 0
        
        self.process_groups_dict = {}
        self.proc_groups_inner_list = []
        self.proc_groups_outer_list = []
        hoop_world_pose_position, hoop_world_pose_orientation = self.obj_11_station_0_revolution.get_local_poses()

        '''side table state'''
        self.depot_state_dic = {0: "empty", 1:"placing", 2: "placed"}
        # self.table_capacity = 4
        self.depot_hoop_set = set()
        self.depot_bending_tube_set = set()
        self.state_depot_hoop = 0
        self.state_depot_bending_tube = 0
        self.depot_product_set = set()

        '''for humans workers (characters) and robots (agv+boxs)'''
        self.character_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Characters/male_adult_construction_01",
            name="character_1",
            track_contact_forces=True,
        )
        self.character_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Characters/male_adult_construction_02",
            name="character_2",
            track_contact_forces=True,
        )
        scene.add(self.character_1)
        scene.add(self.character_2)
        character_list = [self.character_1, self.character_2]
        # self.characters = Characters(character_list)

        self.box_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/AGVs/box_01",
            name="box_1",
            track_contact_forces=True,
        )
        self.box_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/AGVs/box_02",
            name="box_2",
            track_contact_forces=True,
        )
        scene.add(self.box_1)
        scene.add(self.box_2)
        box_list = [self.box_1, self.box_2]
        # self.transboxs = TransBoxs(box_list)

        self.agv_1 = ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/AGVs/agv_01",
            name="agv_1",
            reset_xform_properties=False,
        )
        self.agv_2 = ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/AGVs/agv_02",
            name="agv_2",
            reset_xform_properties=False,
        )
        scene.add(self.agv_1)
        scene.add(self.agv_2)
        agv_list = [self.agv_1, self.agv_2]
        # self.agvs = Agvs(agv_list)

        self.task_manager : TaskManager = TaskManager(character_list, agv_list, box_list)
        '''Ending: for humans workers (characters) and robots (agv+boxs)'''
        # from omniisaacgymenvs.robots.omni_anim_people.scripts.character_behavior import CharacterBehavior
        # from pxr import Sdf
        # prim_path = Sdf.Path(f"/World/envs/env_0" + "/obj/Characters/male_adult_construction_05/ManRoot")
        # self.character_0 = CharacterBehavior(prim_path)
        
        # self.character_0.read_commands_from_file()
        # self.upper_tube_stationt_state_dic = {0:"is_not_full", 1:"fulled"}
        # self.station_state_tube_inner = 0

        # _, hoop_world_pose_orientation = self.materials.hoop_list[self.materials.inner_hoop_processing_index].get_world_poses()
        # from pxr import Gf, UsdGeom
        # self.inital_inner_revolution_matrix = Gf.Matrix4d()
        # position = hoop_world_pose_position.cpu()[0]
        # self.inital_inner_revolution_matrix.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        # orientation = hoop_world_pose_orientation.cpu()[0]
        # self.inital_inner_revolution_matrix.SetRotateOnly(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
        # prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/part9/manipulator2/robotiq_arg2f_base_link")
        # prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/part11/node/Station0")
        # matrix = get_world_transform_matrix(prim)
        # translate = matrix.ExtractTranslation()
        # rotation: Gf.Rotation = matrix.ExtractRotation()
        self.pre_progress_buf = 0
        self.cuda_device = torch.device("cuda:0")
        self.initialize_pre_def_routes(from_file = True)
        return
    
    def post_next_group_to_be_processed_step(self):
        cube_index = self.materials.find_next_raw_cube_index()
        upper_tube_index = self.materials.find_next_raw_upper_tube_index()
        hoop_index = self.materials.find_next_raw_hoop_index()
        bending_tube_index = self.materials.find_next_raw_bending_tube_index()
        #todo find a way to better choose weld station 
        # station_inner_available = self.station_state_inner_middle <=0 and self.station_state_inner_left<=0 and self.station_state_inner_right<=0
        # station_outer_available = self.station_state_outer_middle <=0 and self.station_state_outer_left<=0 and self.station_state_outer_right<=0
        if cube_index<0 or upper_tube_index<0 or hoop_index<0 or bending_tube_index<0:
            return -1, -1, -1, -1
        self.materials.cube_states[cube_index] = 1
        self.materials.hoop_states[hoop_index] = 3
        self.materials.bending_tube_states[bending_tube_index] = 3
        self.materials.upper_tube_states[upper_tube_index] = 1
        _dict = {'cube_index':cube_index, 'upper_tube_index':upper_tube_index,  'hoop_index':hoop_index, 'bending_tube_index':bending_tube_index}
        if len(self.proc_groups_inner_list)<=len(self.proc_groups_outer_list):
            _dict['station'] = 'inner'
            self.proc_groups_inner_list.append(cube_index)
        else:
            self.proc_groups_outer_list.append(cube_index)
            _dict['station'] = 'outer'
        self.process_groups_dict[cube_index] = _dict
        return cube_index, upper_tube_index, hoop_index, bending_tube_index
    
    def initialize_pre_def_routes(self, from_file = False):
        have_problem_routes_character = {
            'put_hoop_into_box':{'hoop_loading_inner':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]], 'hoop_loading_outer':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]],
                'bending_tube_loading_inner':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'bending_tube_loading_outer':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]],
                'cutting_cube': [[-22.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]],  'initial_pose_1': [[-22.0, 14.0, np.deg2rad(180)], [-32.0, 12.0, np.deg2rad(-90)]]}, 
            'put_bending_tube_into_box':{'hoop_loading_inner':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]], 'hoop_loading_outer':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]],
                'bending_tube_loading_inner':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'bending_tube_loading_outer':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]],                             
                'cutting_cube': [[-22.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]],  'initial_pose_1': [[-22.0, 14.0, np.deg2rad(180)], [-32.0, 12.0, np.deg2rad(-90)]]}, 
            'put_hoop_on_table': {'put_bending_tube_on_table':[[-12.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)]], 
                'hoop_loading_inner':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]], 'hoop_loading_outer':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]],
                'bending_tube_loading_inner':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'bending_tube_loading_outer':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]],                             
                'cutting_cube': [[-22.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]],  'initial_pose_1': [[-22.0, 14.0, np.deg2rad(180)], [-32.0, 12.0, np.deg2rad(-90)]]},
            'put_bending_tube_on_table':{'put_hoop_on_table':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]],
                'hoop_loading_inner': [[-28.7, 12.0, np.deg2rad(45)],  [-22.0, 14.0, 0], [-16.26, 14.0, np.deg2rad(-90)]], 'hoop_loading_outer': [[-28.7, 12.0, np.deg2rad(45)],  [-22.0, 14.0, 0], [-16.26, 14.0, np.deg2rad(-90)]],
                'cutting_cube': [[-30.0, 7.0, np.deg2rad(-90)]], 'initial_pose_0': [[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-11.58, 10.0, np.deg2rad(-90)]]},
            'hoop_loading_inner': {'put_hoop_into_box': [[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'put_bending_tube_into_box':[[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 
                'put_hoop_on_table':[[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'put_bending_tube_on_table':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)]],
                'bending_tube_loading_inner':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'bending_tube_loading_outer': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]],
                'cutting_cube': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]], 'placing_product':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)]],
                'initial_pose_0': [[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'initial_pose_1': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-32.0, 12.0, np.deg2rad(-90)]]},
            'bending_tube_loading_inner': {'put_hoop_into_box':[[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 'put_bending_tube_into_box': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 
                'put_hoop_on_table': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]], 'hoop_loading_inner': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]],
                'hoop_loading_outer': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]], 'initial_pose_0': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]]},
            'hoop_loading_outer': {'put_hoop_into_box': [[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'put_bending_tube_into_box':[[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 
                'put_hoop_on_table':[[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'put_bending_tube_on_table':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)]],
                'bending_tube_loading_inner':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'bending_tube_loading_outer': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]],
                'cutting_cube': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]], 'placing_product':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)]],
                'initial_pose_0': [[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'initial_pose_1': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-32.0, 12.0, np.deg2rad(-90)]]},
            'bending_tube_loading_outer': {'put_hoop_into_box':[[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 'put_bending_tube_into_box': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 
                'put_hoop_on_table': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]], 'hoop_loading_inner': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]],
                'hoop_loading_outer': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]], 'initial_pose_0': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]]},
            'cutting_cube': {'put_hoop_into_box':[[-30, 10, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 'put_bending_tube_into_box': [[-30, 10, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]],
                'put_hoop_on_table': [[-30, 10, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]], 'put_bending_tube_on_table': [[-30, 8, np.deg2rad(90)]],
                'hoop_loading_inner':[[-30, 10, np.deg2rad(90)], [-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]], 'hoop_loading_outer':[[-30, 10, np.deg2rad(90)], [-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]],
                'initial_pose_0':[[-30, 10, np.deg2rad(90)], [-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]], 'initial_pose_1':[[-30, 8, np.deg2rad(90)]]},
            'placing_product': {'put_hoop_into_box': [[-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 'put_bending_tube_into_box':[[-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 
                'put_hoop_on_table':[[-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(0)]], 'initial_pose_0': [[-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(0)]],                
                'hoop_loading_inner':[[-22.0, 14.0, np.deg2rad(0)], [-16.26, 14.0, np.deg2rad(-90)]], 'hoop_loading_outer':[[-22.0, 14.0, np.deg2rad(0)], [-16.26, 14.0, np.deg2rad(-90)]]},
            'initial_pose_0' : {'put_bending_tube_on_table': [[-11.58, 10.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-28.7, 12.0, np.deg2rad(-135)]], 'hoop_loading_inner':[[-12.0, 7.4, np.deg2rad(180)], [-16.0, 7.2, np.deg2rad(180)]],  
                'bending_tube_loading_inner':[[-12.0, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'hoop_loading_outer':[[-12.0, 7.4, np.deg2rad(180)], [-16.0, 7.2, np.deg2rad(180)]], 
                'bending_tube_loading_outer':[[-12.0, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'cutting_cube':[[-12.0, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)] , [-26.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]], 
                'placing_product': [[-12.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)]], 'initial_pose_1':[[-11.58, 10.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-28.7, 12.0, np.deg2rad(-135)]]},
            'initial_pose_1': {'put_hoop_into_box': [[-30.0, 12.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 'put_bending_tube_into_box':[[-30.0, 12.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]],
                              'put_hoop_on_table':[[-30.0, 12.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(0)]], 'hoop_loading_inner':[[-30.0, 12.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.26, 14.0, np.deg2rad(-90)]],
                              'hoop_loading_outer':[[-30.0, 12.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.26, 14.0, np.deg2rad(-90)]], 'cutting_cube':[[-30, 8, np.deg2rad(-90)]],
                               'initial_pose_0': [[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, np.deg2rad(0)], [-11.58, 10.0, np.deg2rad(-90)]]}
            }

        have_problem_routes_agv = {
            'carry_box_to_hoop_table':
                {'carry_box_to_bending_tube_table': [[-12.0, 14.0, np.deg2rad(180)], [-22.0, 14.0, np.deg2rad(180)]], 'placing_product': [[-12.0, 14.0, np.deg2rad(180)], [-22.0, 14.0, np.deg2rad(180)]]},
            'carry_box_to_bending_tube_table': {'carry_box_to_hoop_table':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]], 
                'initial_pose_0':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]], 'initial_pose_1':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]],
                'initial_box_pose_0':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]], 'initial_box_pose_1':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]]},
            'placing_product':{'carry_box_to_hoop_table':[[-22.0, 14.0, 0], [-12.4, 14.0, 0]], 
                'initial_pose_0':[[-22.0, 14.0, 0], [-12.4, 14.0, 0]], 'initial_pose_1':[[-22.0, 14.0, 0], [-12.4, 14.0, 0]],
                'initial_box_pose_0':[[-22.0, 14.0, 0], [-12.4, 14.0, 0]], 'initial_box_pose_1':[[-22.0, 14.0, 0], [-12.4, 14.0, 0]]},
        }
        if from_file == True:
            with open(os.path.expanduser(self.cfg_env.env.route_character_file_path), 'rb') as f:
                self.task_manager.characters.routes_dic = pickle.load(f)
            with open(os.path.expanduser(self.cfg_env.env.route_agv_file_path), 'rb') as f:
                self.task_manager.agvs.routes_dic = pickle.load(f)
        else:
            self.xyResolution = 5
            self.obstacleX, self.obstacleY = hybridAStar.map_png(self.xyResolution)
            self.planning_mid_point = [140, 220, 0]
            self.mapParameters = hybridAStar.calculateMapParameters(self.obstacleX, self.obstacleY, self.xyResolution, np.deg2rad(15.0))
            self.task_manager.characters.routes_dic = self.generate_routes(self.task_manager.characters.poses_dic, os.path.expanduser(self.cfg_env.env.route_character_file_path), have_problem_routes_character)
            self.task_manager.agvs.routes_dic = self.generate_routes(self.task_manager.agvs.poses_dic, os.path.expanduser(self.cfg_env.env.route_agv_file_path), have_problem_routes_agv)


    
    def generate_routes(self, pose_dic : dict, file_path, have_problem_routes: dict):
        path = os.path.expanduser(file_path)
        routes_dic = {}
        with open(path, 'rb') as f:
            routes_dic = pickle.load(f)
        for (key, s) in pose_dic.items():
            if key in routes_dic.keys():
                route_dic = routes_dic[key]
            else:
                route_dic = {}
            for (_key, g) in pose_dic.items():
                if _key == key or _key in route_dic.keys():
                    continue
                if key in have_problem_routes.keys() and _key in have_problem_routes[key].keys():
                    x, y, yaw = self.path_planner_multi_poses(s.copy(), g.copy(), have_problem_routes[key][_key].copy())
                else:
                    x, y, yaw = self.path_planner(s.copy(), g.copy())
                route_dic[_key] = (x,y,yaw)
            routes_dic[key] = route_dic
            with open(path, 'wb') as f:
                pickle.dump(routes_dic, f)
        return


    def path_planner_multi_poses(self, start, goal, interval_path_list):
        interval_path_list = [start] + interval_path_list + [goal]
        x= []
        y = []
        yaw = []
        trans_x = -50
        trans_y = -30
        for i in range(0, len(interval_path_list) - 1):
            s = interval_path_list[i].copy()
            g = interval_path_list[i+1].copy()
            _x, _y, _yaw = self.path_planner(s, g)
            x += _x
            y += _y
            yaw += _yaw
        _x = [(value - trans_x)*self.xyResolution for value in x]
        _y = [(value - trans_y)*self.xyResolution for value in y]
        visualize = False
        if visualize:
            import math
            for k in range(len(_x)):
                plt.cla()
                plt.xlim(min(self.obstacleX), max(self.obstacleX)) 
                plt.ylim(min(self.obstacleY), max(self.obstacleY))        
                # plt.xlim(x_limit[0], x_limit[1]) 
                # plt.ylim(y_limit[0], y_limit[1])                
                plt.xlim(0, 300) 
                plt.ylim(0, 250)
                plt.plot(self.obstacleX, self.obstacleY, "sk")
                # plt.plot(s, g, linewidth=1.5, color='r', zorder=0)
                plt.plot(_x, _y, linewidth=1.5, color='r', zorder=0)
                # hybridAStar.drawCar(s[0], s[1], s[2])
                # hybridAStar.drawCar(g[0], g[1], g[2])
                hybridAStar.drawCar(_x[k], _y[k], yaw[k])
                plt.arrow(_x[k], _y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
                plt.title("Hybrid A*")
                plt.pause(0.01)
        return x, y, yaw

    def scale_pose(self, _pose: list):
        pose = _pose.copy()
        trans_x = -50
        trans_y = -30
        pose[0] = (pose[0]/self.xyResolution + trans_x)
        pose[1] = (pose[1]/self.xyResolution + trans_y)
        return pose

    def path_planner(self, s, g):
        dis = np.linalg.norm(np.array(s[:2]) - np.array(g[:2]))
        if dis < 0.1:
            x, y, yaw = [s[0], g[0]], [s[1], g[1]], [s[2], g[2]]
            return x,y,yaw

        # self.xyResolution = 5
        trans_x = -50
        trans_y = -30
        # Set Start, Goal x, y, theta
        # s = [0, 10, np.deg2rad(90)]
        # g = [-13.3, 6, np.deg2rad(90)]
        s[0] = (s[0] - trans_x)*self.xyResolution
        s[1] = (s[1] - trans_y)*self.xyResolution
        g[0] = (g[0] - trans_x)*self.xyResolution
        g[1] = (g[1] - trans_y)*self.xyResolution
        # self.obstacleX, self.obstacleY = hybridAStar.map_png(self.xyResolution)
        # # Calculate map Paramaters
        # self.mapParameters = hybridAStar.calculateself.MapParameters(self.obstacleX, self.obstacleY, self.xyResolution, np.deg2rad(15.0))
        # Run Hybrid A*
        dis_s_m = np.linalg.norm(np.array(s) - np.array(self.planning_mid_point))
        dis_g_m = np.linalg.norm(np.array(g) - np.array(self.planning_mid_point))
        import time  # 引入time模块
        if min(s[0], g[0]) < self.planning_mid_point[0] and self.planning_mid_point[0] < max(s[0], g[0]) and dis_s_m > 10 and dis_g_m > 10:
            self.planning_mid_point[2] = 0 if (g[0] - s[0]) >=0  else np.deg2rad(180)
            start_t = time.time()
            x1, y1, yaw1 = hybridAStar.run(s, self.planning_mid_point, self.mapParameters, plt)
            x2, y2, yaw2 = hybridAStar.run(self.planning_mid_point, g, self.mapParameters, plt)
            end_t = time.time()
            if end_t-start_t > 3.:
                a = 1
            x = x1 + x2[1:]
            y = y1 + y2[1:]
            yaw = yaw1 + yaw2[1:]
            a = 1
        else:
            start_t = time.time()
            x, y, yaw = hybridAStar.run(s, g, self.mapParameters, plt)
            end_t = time.time()
            if end_t-start_t > 3.:
                a = 1
        # x_limit = [min(self.obstacleX), max(self.obstacleX)]
        # y_limit = [min(self.obstacleY), max(self.obstacleY)]
        scale_flag = True
        if scale_flag:
            x = [value/self.xyResolution + trans_x for value in x]
            y = [value/self.xyResolution + trans_y for value in y]
            # self.obstacleX = [value/self.xyResolution + trans_x  for value in self.obstacleX]
            # self.obstacleY = [value/self.xyResolution + trans_y for value in self.obstacleY]
        # # Draw Animated Car
        # import math
        visualize = False
        def show_map_s_g():
            import math
            plt.cla()
            plt.xlim(min(self.obstacleX), max(self.obstacleX)) 
            plt.ylim(min(self.obstacleY), max(self.obstacleY))        
            # plt.xlim(x_limit[0], x_limit[1]) 
            # plt.ylim(y_limit[0], y_limit[1])                
            plt.xlim(0, 300) 
            plt.ylim(0, 250)
            plt.plot(self.obstacleX, self.obstacleY, "sk")
            # plt.plot(s, g, linewidth=1.5, color='r', zorder=0)
            # plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
            hybridAStar.drawCar(s[0], s[1], s[2])
            hybridAStar.drawCar(g[0], g[1], g[2])
            plt.arrow(s[0], s[1], 1*math.cos(s[2]), 1*math.sin(s[2]), width=.1)
            plt.arrow(g[0], g[1], 1*math.cos(g[2]), 1*math.sin(g[2]), width=.1)
            plt.title("Hybrid A*")
            plt.pause(0.01)

        if visualize:
            # x_limit = [-50, 30]
            # y_limit= [-30, 40]
            s[0] = (s[0])/self.xyResolution + trans_x
            g[0] = (g[0])/self.xyResolution + trans_x
            s[1] = (s[1])/self.xyResolution + trans_y
            g[1] = (g[1])/self.xyResolution + trans_y
            import math
            for k in range(len(x)):
                plt.cla()
                plt.xlim(min(self.obstacleX), max(self.obstacleX)) 
                plt.ylim(min(self.obstacleY), max(self.obstacleY))        
                # plt.xlim(x_limit[0], x_limit[1]) 
                # plt.ylim(y_limit[0], y_limit[1])                
                plt.xlim(0, 300) 
                plt.ylim(0, 250)
                plt.plot(self.obstacleX, self.obstacleY, "sk")
                # plt.plot(s, g, linewidth=1.5, color='r', zorder=0)
                plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
                # hybridAStar.drawCar(s[0], s[1], s[2])
                # hybridAStar.drawCar(g[0], g[1], g[2])
                hybridAStar.drawCar(x[k], y[k], yaw[k])
                plt.arrow(x[k], y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
                plt.title("Hybrid A*")
                plt.pause(0.01)
        
        # plt.show()
        def sample(x, interval = 2):
            _x = x[1:-1]
            _x = _x[::interval]
            x = [x[0]] + _x + [x[-1]]
            return x
        x, y, yaw = sample(x), sample(y), sample(yaw)
        return x, y, yaw


    def initialize_views(self, scene) -> None:
        """Initialize views for extension workflow."""
        super().initialize_views(scene)
        self._import_env_assets(add_to_stage=False)

    def _import_env_assets(self, add_to_stage=True):
        """Import modular production assets."""

        self.obj_heights = []
        self.obj_widths_max = []
        self.thread_pitches = []
        self._stage = get_current_stage()
        assets_root_path = get_assets_root_path()

        for i in range(0, self._num_envs):
            for j in range(0, len(self.cfg_env.env.desired_subassemblies)):

                subassembly = self.cfg_env.env.desired_subassemblies[j]
                components = list(self.asset_info_obj[subassembly])

                obj_translation = torch.tensor(
                    [
                        i*10,
                        i*10,
                        0,
                    ],
                    device=self._device,
                )
                obj_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

                obj_height = self.asset_info_obj[subassembly][components[0]]["height"]
                obj_width_max = self.asset_info_obj[subassembly][components[0]][
                    "width_max"
                ]
                self.obj_heights.append(obj_height)
                self.obj_widths_max.append(obj_width_max)

                obj_file = (
                    self.asset_info_obj[subassembly][components[0]]["usd_path"]
                )

                if add_to_stage:
                    add_reference_to_stage(usd_path = obj_file, prim_path = f"/World/envs/env_{i}" + "/obj")
                    XFormPrim(
                        prim_path=f"/World/envs/env_{i}" + "/obj",
                        translation=obj_translation,
                        orientation=obj_orientation,
                    )
                    self._sim_config.apply_articulation_settings(
                        "obj",
                        self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj"),
                        self._sim_config.parse_actor_config("obj"),
                    )
                thread_pitch = self.asset_info_obj[subassembly]["thread_pitch"]
                self.thread_pitches.append(thread_pitch)
        # For computing body COM pos
        self.obj_heights = torch.tensor(
            self.obj_heights, device=self._device
        ).unsqueeze(-1)

        # For setting initial state
        self.obj_widths_max = torch.tensor(
            self.obj_widths_max, device=self._device
        ).unsqueeze(-1)

        self.thread_pitches = torch.tensor(
            self.thread_pitches, device=self._device
        ).unsqueeze(-1)

    def refresh_env_tensors(self):
        """Refresh tensors."""
        pass
