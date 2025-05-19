# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch import nn

from rl_games.common import vecenv
from rl_games.algos_torch import torch_ext
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from omniisaacgymenvs.algo.rainbowmini.memory import ReplayMemoryPPO
from omniisaacgymenvs.algo.rainbowmini.model import Actor, Critic
from tqdm import trange
import time
from omegaconf import DictConfig
from omniisaacgymenvs.utils.data import data
import wandb
import copy
import torch.nn.functional as F

class PPOAgent():
    def __init__(self, base_name, params):

        self.config : DictConfig = params['config']
        self.base_init(base_name)
        config = self.config
        print(config)
        #######pramameters for model update and act
        self.Vmin = config.get('V_min', -10)
        self.Vmax = config.get('V_max', 10)

        self.n = config['multi_step']
        self.discount = config['discount']
        self.norm_clip = config.get('norm_clip', 10)
        ###########for agent training
        self.update_frequency = config.get('update_frequency', 200)
        self.evaluate_interval = config.get('evaluate_interval', 100)
        self.start_evaluate_step = int(config["ppo_buffer_size"])
        self.target_update = config.get('target_update', int(2e3))
        self.max_steps = config.get("max_steps", int(7e5))
        self.max_epochs = config.get("max_epochs", int(1e11))
        self.batch_size = config.get('batch_size', 512)
        # self.batch_size = config.get('batch_size', 2)
        self.num_warmup_steps = config.get('num_warmup_steps', int(1e2))
        # self.num_warmup_steps = config.get('num_warmup_steps', int(1024))
        self.demonstration_steps = config.get('demonstration_steps', int(0))
        self.num_steps_per_epoch = config.get("num_steps_per_epoch", 100)
        self.max_env_steps = config.get("horizon_length", 1000) # temporary, in future we will use other approach
        print(self.batch_size, self.num_actors, self.num_agents)
        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)
        #########buffer
        self.priority_weight_increase = (1 - config['priority_weight']) / (self.max_steps - self.num_warmup_steps)
        self.replay_buffer = ReplayMemoryPPO(config, int(config["ppo_buffer_size"]))
        # self.ppo_buffer_size = config["ppo_buffer_size"]
        self.ppo_each_train_epochs = config.get('ppo_each_train_epochs', 10)
        ####### actor critic
        self.actor = Actor(config, self.actions_num).to(device=self._device)
        self.critic = Critic(config, self.actions_num).to(device=self._device)
        if self._test and not self.env_rule_based_exploration:
            weights = torch.load(self.train_dir + self._load_dir + self._load_name, weights_only=True)
            self.actor.load_state_dict(weights['actor'])
            self.critic.load_state_dict(weights['critic'])
        self.actor.train()
        self.critic.train()
        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps = 0.2
        #####
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=config['learning_rate'], eps=config['adam_eps'])
        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=config['learning_rate'], eps=config['adam_eps'])
        # self.loss_criterion = nn.MSELoss(reduction= 'none')
        self.use_wandb = config.get('wandb_activate', False)
        if self.use_wandb:
            self.init_wandb_logger()
    # def load_networks(self, params):
    #     builder = model_builder.ModelBuilder()
    #     config['network'] = builder.load(params)
    def setdefault(self, dict: dict, key, default):
        if key in dict:
            return
        else:
            dict.__setitem__(key, default)

    def base_init(self, base_name):

        self.setdefault(self.config, key='device', default='cuda:0')
        ########for replay buffer args initialize
        self.setdefault(self.config, key='replay_buffer_size', default=int(5e5))
        self.setdefault(self.config, key='history_length', default=1)
        self.setdefault(self.config, key='discount', default=0.99)
        self.setdefault(self.config, key='multi_step', default=1)
        self.setdefault(self.config, key='priority_exponent', default=0.5)
        self.setdefault(self.config, key='priority_weight', default=0.4)
        ########for neural network args initialize
        self.setdefault(self.config, key='architecture', default='canonical')
        self.setdefault(self.config, key='hidden_size', default=512)
        self.setdefault(self.config, key='noisy_std', default=0.1)
        ######for optimizer initialize
        # self.setdefault(self.config, key='learning_rate', default=0.0000625)
        self.setdefault(self.config, key='learning_rate', default=1e-4)
        self.setdefault(self.config, key='adam_eps', default=1.5e-4)
        config = self.config
        ####TODO
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self.action_space = self.env_info['action_space']
        self.actions_num = self.action_space.shape[0]
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]
        self.observation_space = self.env_info['observation_space']
        self.obs_shape = self.observation_space.shape
        self.obs = None
        self._device = config['device']
        ##test
        self._test = config['test']
        self._load_dir = config['load_dir']
        self._load_name = config['load_name']
        self.env_rule_based_exploration = config.get('env_rule_based_exploration', False)
        #temporary for Isaac gym compatibility
        print('Env info:')
        print(self.env_info)

        # self.rewards_shaper = config['reward_shaper']
        
        # self.weight_decay = config.get('weight_decay', 0.0)
        #self.use_action_masks = config.get('use_action_masks', False)
        # self.is_train = config.get('is_train', True)
        # self.c_loss = nn.MSELoss()
        # self.c2_loss = nn.SmoothL1Loss()

        self.save_best_after = config.get('save_best_after', 500)
        # self.print_stats = config.get('print_stats', True)
        # self.rnn_states = None
        # self.name = base_name

        self.save_freq = config.get('save_frequency', 0)

        # self.network = config['network']
        self.num_agents = self.env_info.get('agents', 1)

        self.games_to_track = config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.train_avgs = []
        self.eval_avgs = []
        for i in range(0, self.config['max_num_worker']):
            list_a = []
            list_b = []
            for j in range(0, self.config['max_num_robot']):
                list_a.append(torch_ext.AverageMeter(1, self.games_to_track).to(self._device))
                list_b.append(torch_ext.AverageMeter(1, self.games_to_track).to(self._device))
            self.train_avgs.append(list_a)
            self.eval_avgs.append(list_b)
    
        # self.min_alpha = torch.tensor(np.log(1)).float().to(self._device)
        self.step_num = 0
        self.epoch_num = 0
        self.episode_num = 0
        self.update_time = 0
        self.last_mean_rewards = -1000000000
        self.play_time = 0
        
        self.evaluate_step_num = 0
        self.evaluate_episode_num = 0
        # TODO: put it into the separate class
        pbt_str = '_'
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        # full_experiment_name = config.get('full_experiment_name', None)
        # if full_experiment_name:
        #     print(f'Exact experiment name requested from command line: {full_experiment_name}')
        #     self.experiment_name = full_experiment_name
        # else:
        #     self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
        # time_now = datetime.now().strftime("_%d-%H-%M-%S")
        time_now = self.config['time_str']
        if not self._test:
            self.experiment_name = config['name'] + pbt_str + time_now
        else:
            if self.env_rule_based_exploration:
                self.experiment_name = 'test_rule'+ pbt_str + time_now
            else:
                self.experiment_name = 'test'+ '_'.join(self._load_name.split('_')[1:3]) + '_' + self._load_dir[-22:-3]
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter('runs/' + config['name'] + time_now)
        print("Run Directory:", config['name'] + time_now)

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_wandb_logger(self):
        wandb.define_metric("Train/step")
        wandb.define_metric("Train/buffer_size", step_metric="Train/step")
        wandb.define_metric("Train/actor_loss", step_metric="Train/step")
        wandb.define_metric("Train/critic_loss", step_metric="Train/step")
        wandb.define_metric("Train/train_epoch", step_metric="Train/step")

        wandb.define_metric("Train/Mrewards", step_metric="Train/step")
        wandb.define_metric("Train/MLen", step_metric="Train/step")
        wandb.define_metric("Metrics/step_episode", step_metric="Train/step")
        wandb.define_metric("Metrics/EpRet", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpLen", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpEnvLen", step_metric="Metrics/step_episode")

        wandb.define_metric("Metrics/EpTime", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpProgress", step_metric="Metrics/step_episode")
        wandb.define_metric("Metrics/EpRetAction", step_metric="Metrics/step_episode")

 
        total = sum([param.nelement() for param in self.actor.parameters()] + [param.nelement() for param in self.critic.parameters()])
        # print("Number of parameters: %.2fM" % (total/1e6))
        param_table = wandb.Table(columns=["online_net_size", "num_warm_up_steps"], data=[[total, self.num_warmup_steps]])
        wandb.log({"Parameter": param_table})

        #evaluate
        wandb.define_metric("Evaluate/step")
        wandb.define_metric("Evaluate/step_episode", step_metric="Evaluate/step")
        wandb.define_metric("Evaluate/EpRet", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpLen", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpEnvLen", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpTime", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpProgress", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/EpRetAction", step_metric="Evaluate/step_episode")
        wandb.define_metric("Evaluate/Savepth", step_metric="Evaluate/step_episode")
        for i in range(0, self.config['max_num_worker']):
            for j in range(0, self.config['max_num_robot']):
                wandb.define_metric(f'Avg/{i+1}_{j+1}', step_metric="Evaluate/step")
        # self.evaluate_table = wandb.Table(columns=["env_length", "action_seq", "progress"])

        #test
        self.test_table = wandb.Table(columns=["worker_initial_pose", "robot_initial_pose", "box_initial_pose", "progress", "env_length"])
        self.test_table2 = wandb.Table(columns=["num_worker", "num_robot&box", "max", "min", "mean"])
        self.test_table3 = wandb.Table(columns=["time_step", "action_list"])
        return
    
    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.actor.reset_noise()
        self.critic.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            prob = self.actor(data.func(state, 'unsqueeze', 0))
            action = prob.argmax(1)
            return action.unsqueeze(0), prob
            # return (self.online_net(data.func(state, 'unsqueeze', 0)) * self.support).sum(2)

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return self.act_random(state) if np.random.random() < epsilon else self.act(state)
    
    def act_random(self, state):
        action_mask = state['action_mask']
        indexs = action_mask.nonzero()
        index = torch.randint(low=0, high = len(indexs), size = (1,), device=self._device) 
        action = indexs[index]
        return action

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def set_train(self):
        self.actor.train()
        self.critic.train()

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors
        self.temp_current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)
        self.temp_dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_rewards_action = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)
        self.current_ep_time = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)        
        
        self.evaluate_current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.evaluate_current_rewards_action = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.evaluate_current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)
        self.evaluate_current_ep_time = torch.zeros(batch_size, dtype=torch.float32, device=self._device)

        self.count_task_times = torch.zeros([self.config["max_num_worker"], self.config["max_num_robot"]], dtype=torch.float32, device=self._device)
        self.count_task_success = torch.zeros([self.config["max_num_worker"], self.config["max_num_robot"]], dtype=torch.float32, device=self._device)
        self.task_succ_rate = torch.zeros([self.config["max_num_worker"], self.config["max_num_robot"]], dtype=torch.float32, device=self._device)

    @property
    def device(self):
        return self._device

    def get_weights(self):
        print("Loading weights")
        state = {'actor':self.actor.state_dict(), 'critic':self.critic.state_dict()}
        return state

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.online_net.load_state_dict(weights['actor'])

    def get_full_state_weights(self):
        print("Loading full weights")
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        # state['optimizer'] = self.optimiser.state_dict()       

        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)

        if set_epoch:
            self.epoch_num = weights['epoch']

        self.optimiser.load_state_dict(weights['optimizer'])
        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def restore(self, fn, set_epoch=True):
        print("rainbow restore")
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1.0 - tau) * target_param.data)

    def update(self, step):
        # Sample transitions
        actor_loss_list = []
        critic_loss_list = []
        for i in range(self.ppo_each_train_epochs):
            idxs, states, actions, actions_prob, returns, next_states, nonterminals, weights = self.replay_buffer.sample(self.batch_size)
            states = data.stack_from_array(states.squeeze(), device=self._device)
            next_states = data.stack_from_array(next_states.squeeze(), device=self._device)

            with torch.no_grad():
                actions_prob = data.stack_from_array(actions_prob.squeeze(), device=self._device)['action_prob']
                old_log_probs = torch.log(actions_prob.gather(1, actions.unsqueeze(-1)))
                terminals = torch.where(nonterminals == 1.0, 0., 1.0)
                td_target = returns.unsqueeze(-1) + self.gamma * self.critic(next_states) * (1 - terminals)
                td_delta = td_target - self.critic(states)
                advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self._device)

            log_probs = torch.log(self.actor(states).gather(1, actions.unsqueeze(-1)))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimiser.zero_grad()
            self.critic_optimiser.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimiser.step()
            self.critic_optimiser.step()

            self.replay_buffer.update_priorities(idxs, critic_loss.detach().cpu().numpy())  # Update priorities of sampled transitions

            if self.use_wandb:
                wandb.log({
                        'Train/step': self.step_num,
                        "Train/actor_loss": actor_loss.mean().item(),
                        "Train/critic_loss": critic_loss.mean().item(),
                    })
            time_now = datetime.now().strftime("_%d-%H-%M-%S")   
            print("time_now:{}".format(time_now) +" traning actor_loss:{}, critic_loss:{}".format(actor_loss.mean().item(), critic_loss.mean().item()))
            actor_loss_list.append(actor_loss.detach().item())
            critic_loss_list.append(critic_loss.detach().item())
        self.replay_buffer.clear_data()
        return actor_loss_list, critic_loss_list

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage.item())
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)


    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
            obs = obs.to(self._device)
        elif isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self._device)
        return obs

    # TODO: move to common utils
    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        # if not obs_is_dict or 'obs' not in obs:    
        #     upd_obs = {'obs' : upd_obs}

        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)

        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        return actions

    def env_step(self, actions):
        # actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, actions = self.vec_env.step(actions) # (obs_space) -> (n, obs_space)

        return self.obs_to_tensors(obs), rewards.to(self._device), dones.to(self._device), infos, actions


    def env_reset(self, num_worker=None, num_robot=None):
        with torch.no_grad():
            obs = self.vec_env.reset(num_worker, num_robot)

        obs = self.obs_to_tensors(obs)

        return obs

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1000000000
    
    def train_epoch(self, num_worker=None, num_robot=None):
        temporary_buffer, reward_extra, repeat_times = self.play_steps(num_worker, num_robot)
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        loss = None
        for j in range(repeat_times):
            for i in range(len(temporary_buffer)):
                random_exploration = self.step_num < self.num_warmup_steps
                self.set_train()
                if self.step_num % self.update_frequency == 0:
                    self.reset_noise()
                #debug TODO
                # action = None
                step_start = time.time()
                obs, action, action_prob, rewards, dones, infos = temporary_buffer[i]
                # if self.reward_clip > 0:
                #     reward = max(min(reward, self.reward_clip), -self.reward_clip)  # Clip rewards
                step_end = time.time()
                #TODO only support num_agents == 1
                assert self.num_agents == 1, ('only support num_agents == 1')
                self.step_num += self.num_actors * 1
                self.current_rewards += rewards+reward_extra
                self.current_rewards_action += infos["rew_action"]
                self.current_lengths += 1
                self.current_ep_time += (step_end - step_start)
                total_time += (step_end - step_start)
                step_time += (step_end - step_start)

                all_done_indices = dones.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                self.game_rewards.update(self.current_rewards[done_indices])
                self.game_lengths.update(self.current_lengths[done_indices])

                # no_timeouts = self.current_lengths <= self.max_env_steps
                # dones = dones * no_timeouts
                not_dones = 1.0 - dones.float()
                obs_cpu = {}
                for key, value in obs.items():
                    obs_cpu[key] = value.cpu()
                action_cpu = action.squeeze().cpu()
                action_prob_cpu = action_prob.squeeze().cpu()
                rewards_cpu = rewards.squeeze().cpu()
                dones_cpu = dones.squeeze().cpu()
                self.replay_buffer.append(obs_cpu, action_cpu, {'action_prob': action_prob_cpu}, rewards_cpu+reward_extra, dones_cpu)

                if dones[0]:
                    self.episode_num += 1
                    if self.use_wandb:
                        wandb.log({
                            "Train/step": self.step_num,
                            'Metrics/step_episode': self.episode_num,
                            'Metrics/EpRet': self.current_rewards,
                            'Metrics/EpLen': self.current_lengths,
                            'Metrics/EpEnvLen': infos['env_length'],
                            "Metrics/EpTime": self.current_ep_time,
                            "Metrics/EpProgress": infos['progress'],
                            "Metrics/EpRetAction": self.current_rewards_action,
                        })
                    # next_obs = self.env_reset()   
                    if self.step_num > self.start_evaluate_step and self.episode_num % self.evaluate_interval == 0:
                        #TODO debug
                        # pass
                        success_list= []
                        w, r = 1, 1
                        self.obs = self.env_reset(1,1)
                        for _i in range(self.config["max_num_worker"]):
                            for _j in range(self.config["max_num_robot"]):
                                r += 1
                                if r>self.config["max_num_robot"]:
                                    r = 1
                                    w += 1
                                if w>self.config["max_num_worker"]:
                                    w, r = None, None
                                success_list.append(self.evaluate_epoch(test=False, reset_n_worker=w, reset_n_robot=r))
                        if np.all(success_list):
                            # checkpoint_name = self.config['name'] + '_ep_' + str(self.episode_num) + '_len_' + str(infos['env_length'].item()) + '_rew_' + "{:.2f}".format(self.evaluate_current_rewards.item())
                            checkpoint_name = self.config['name'] + '_ep_' + str(self.episode_num)
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            if self.use_wandb:
                                wandb.log({"Evaluate/Savepth": self.episode_num,
                                })
                self.current_rewards = self.current_rewards * not_dones
                self.current_lengths = self.current_lengths * not_dones
                self.current_ep_time = self.current_ep_time * not_dones
                self.current_rewards_action = self.current_rewards_action * not_dones

                if not random_exploration:
                    # self.replay_buffer.priority_weight = min(self.replay_buffer.priority_weight + self.priority_weight_increase, 1)
                    if self.replay_buffer.transitions.full:
                        self.set_train()
                        actor_loss_list, critic_loss_list = self.update(self.epoch_num)

            total_time_end = time.time()
            total_time = total_time_end - total_time_start
            play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, loss
    
    def play_steps(self, num_worker=None, num_robot=None):
        temporary_buffer = []
        while True:
            obs : dict = self.obs
            random_exploration = self.step_num < self.num_warmup_steps
            self.set_train()
            # if random_exploration:
            #     if self.step_num < self.demonstration_steps:
            #         action = None
            #     else: 
            #         action = self.act_random(obs)
            #         # action_mask = obs['action_mask']
            #         # indexs = action_mask.nonzero()
            #         # index = torch.randint(low=0, high = len(indexs), size = (1,), device=self._device) 
            #         # action = indexs[index]
            # else:
            #     with torch.no_grad():
            #         action, action_prob = self.act(obs).unsqueeze(0)
            #         # action = self.act_e_greedy(obs).unsqueeze(0)
            action, action_prob = self.act(obs)
            #debug TODO
            # action = None

            with torch.no_grad():
                next_obs, rewards, dones, infos, action = self.env_step(action)
            # if self.reward_clip > 0:
            #     reward = max(min(reward, self.reward_clip), -self.reward_clip)  # Clip rewards
            #TODO only support num_agents == 1
            assert self.num_agents == 1, ('only support num_agents == 1')
            self.temp_current_lengths += 1

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            # no_timeouts = self.temp_current_lengths <= self.max_env_steps
            # dones = dones * no_timeouts
            not_dones = 1.0 - dones.float()

            self.temp_current_lengths = self.temp_current_lengths * not_dones
            # obs_copy = {}
            # infos_copy = {}
            # for key, value in obs.items():
            #     obs_copy[key] = value.copy()
            # for key, value in infos.items():
            #     infos_copy[key] = value.copy()
            # action_cpu = action.squeeze().cpu()
            # rewards_cpu = rewards.squeeze().cpu()
            # dones_cpu = dones.squeeze().cpu()
            temporary_buffer.append((copy.deepcopy(obs), copy.deepcopy(action), copy.deepcopy(action_prob), copy.deepcopy(rewards), copy.deepcopy(dones), copy.deepcopy(infos)))
            if dones[0]:
                next_obs = self.env_reset(num_worker=num_worker, num_robot=num_robot)   
            self.obs = next_obs.copy()
            reward_extra = 0.
            repeat_times = 1
            if dones[0]:
                _,_,_,_,_, _infos = temporary_buffer[-1]
                goal_finished = _infos['env_length'] < _infos['max_env_len']-1 and _infos['progress'] == 1
                if goal_finished:
                    reward_extra = 0.4*(_infos['max_env_len']-1 - _infos['env_length'])/_infos['env_length']
                    repeat_times = 5
                else:
                    if len(temporary_buffer) < 100:
                        reward_extra = -0.05
                    else:
                        reward_extra = -0.001
                # print("reward_extra:{}, env_len:{}".format(reward_extra, _infos['env_length']))
                if not random_exploration or goal_finished:
                    break
        return temporary_buffer, reward_extra, repeat_times
    
    def evaluate_epoch(self, test=False, reset_n_worker=None, reset_n_robot=None):
        total_time_start = time.time()
        total_time = 0
        step_time = 0.0
        action_info_list = []
        task_success = False
        if test:
            time_step_list = []
        while True:
            self.set_eval()
            obs : dict = self.obs
            if self.env_rule_based_exploration:
                action = None
            else:
                with torch.no_grad():
                    action, action_prob = self.act(obs)

            step_start = time.time()
            with torch.no_grad():
                next_obs, rewards, dones, infos, action = self.env_step(action)
            # if self.reward_clip > 0:
            #     reward = max(min(reward, self.reward_clip), -self.reward_clip)  # Clip rewards
            step_end = time.time()
            #TODO only support num_agents == 1
            assert self.num_agents == 1, ('only support num_agents == 1')
            self.evaluate_step_num += self.num_actors * 1
            self.evaluate_current_rewards += rewards
            self.evaluate_current_rewards_action += infos["rew_action"]
            action_info_list.append(infos["action_info"])
            if test:
                time_step_list.append(infos["time_step"])
            self.evaluate_current_lengths += 1
            self.evaluate_current_ep_time += (step_end - step_start)
            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            # no_timeouts = self.evaluate_current_lengths != self.max_env_steps
            # dones = dones * no_timeouts
            not_dones = 1.0 - dones.float()            
            if dones[0]:
                if self.use_wandb:
                    wandb.log({
                        'Evaluate/step': self.evaluate_step_num,
                        'Evaluate/step_episode': self.evaluate_episode_num,
                        'Evaluate/EpRet': self.evaluate_current_rewards,
                        'Evaluate/EpEnvLen': infos['env_length'],
                        'Evaluate/EpLen': self.evaluate_current_lengths,
                        "Evaluate/EpTime": self.evaluate_current_ep_time,
                        "Evaluate/EpProgress": infos['progress'],
                        "Evaluate/EpRetAction": self.evaluate_current_rewards_action,
                    })   
                    if infos['env_length'] < infos['max_env_len']-1 and infos['progress'] == 1:
                        task_success = True
                    num_worker, num_robot = infos['num_worker'], infos['num_robot']
                    self.eval_avgs[num_worker-1][num_robot-1].update(torch.tensor([task_success], dtype=torch.float32, device=self._device))
                    wandb.log({f'Avg/{num_worker}_{num_robot}': self.eval_avgs[num_worker-1][num_robot-1].get_mean()}) 
                        # self.evaluate_table.add_data(infos['env_length'], ' '.join(action_info_list), infos['progress'])
                        # wandb.log({"Action": self.evaluate_table}) 
                        # if not test:
                        #     # checkpoint_name = self.config['name'] + '_ep_' + str(self.episode_num) + '_len_' + str(infos['env_length'].item()) + '_rew_' + "{:.2f}".format(self.evaluate_current_rewards.item())
                        #     checkpoint_name = self.config['name'] + '_ep_' + str(self.episode_num)
                        #     self.save(os.path.join(self.nn_dir, checkpoint_name)) 
                    if test:
                        self.test_table.add_data(infos['worker_initial_pose'] , infos["robot_initial_pose"], infos['box_initial_pose'], infos['progress'], infos['env_length'].cpu())
                        self.test_table3.add_data(' '.join(time_step_list), ' '.join(action_info_list))
                action_info_list = []
                print(" evaluate ", end=' ')
                next_obs = self.env_reset(num_worker=reset_n_worker, num_robot=reset_n_robot) 
            self.evaluate_current_rewards = self.evaluate_current_rewards * not_dones
            self.evaluate_current_lengths = self.evaluate_current_lengths * not_dones
            self.evaluate_current_ep_time = self.evaluate_current_ep_time * not_dones
            self.evaluate_current_rewards_action = self.evaluate_current_rewards_action * not_dones
            self.obs = next_obs.copy()
            if dones[0]:
                self.evaluate_episode_num += 1  
                break

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        return task_success
    
    def train(self):
        self.init_tensors()
        total_time = 0
        # rep_count = 0
        self.obs = self.env_reset()
        while True:
            self.epoch_num += 1
            if self._test:     
                for w in range(self.config["max_num_worker"]):
                    for r in range(self.config["max_num_robot"]):
                        for i in range(self.config['test_times']):
                            self.evaluate_epoch(test=True)
                        if self.use_wandb:
                            index = w*self.config["max_num_worker"]+r
                            time_span = self.test_table.get_column("env_length")[index*self.config['test_times']: (index+1)*self.config['test_times']]
                            self.test_table2.add_data(w+1, r+1, np.max(time_span), np.min(time_span), np.mean(time_span))
                if self.use_wandb:
                    wandb.log({"Instances": self.test_table}) 
                    wandb.log({"Instances2": self.test_table2}) 
                    wandb.log({"Actions": self.test_table3}) 
                    wandb.finish()
                break
            else:
                for w in range(self.config["max_num_worker"]):
                    for r in range(self.config["max_num_robot"]):
                        step_time, play_time, update_time, epoch_total_time, loss = self.train_epoch(w+1,r+1)

                        # total_time += epoch_total_time
                        # self.step_num += self.num_steps_per_epoch

                        # fps_step = self.num_steps_per_epoch / step_time
                        # fps_step_inference = self.num_steps_per_epoch / play_time
                        # fps_total = self.num_steps_per_epoch / epoch_total_time

                        if self.use_wandb:
                            wandb.log({
                                    "Train/step": self.step_num,
                                    "Train/train_epoch": self.epoch_num,
                                    'Train/buffer_size': self.replay_buffer.transitions.index,
                                })  
                            if self.game_rewards.current_size > 0:
                                wandb.log({
                                    'Train/Mrewards': self.game_rewards.get_mean(),
                                    'Train/MLen': self.game_lengths.get_mean(),
                                })  
                if self.step_num > self.max_steps:
                    if self.use_wandb:
                        wandb.finish()
                    break


