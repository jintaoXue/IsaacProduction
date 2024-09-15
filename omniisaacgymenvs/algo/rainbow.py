# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch import nn
from model import DQN

from rl_games.common import vecenv
from rl_games.algos_torch import torch_ext
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class RainbowAgent():
    def __init__(self, base_name, params):

        self.config = config = params['config']
        print(config)

        # TODO: Get obs shape and self.network
        # self.load_networks(params)
        self.base_init(base_name, config)

        self.action_space = self.env_info['action_space']
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
        if os.path.isfile(args.model):
            state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
            if 'conv1.weight' in state_dict.keys():
            for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
                del state_dict[old_key]  # Delete old keys for strict load_state_dict
            self.online_net.load_state_dict(state_dict)
            print("Loading pretrained model: " + args.model)
        else:  # Raise error if incorrect model path provided
            raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
        param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

        #############################################################todo
        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = float(config["critic_tau"])
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)
        self.normalize_input = config.get("normalize_input", False)

        # TODO: double-check! To use bootstrap instead?
        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach

        print(self.batch_size, self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self._device)
        self.log_alpha.requires_grad = True
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)

        self.actor_optimizer = torch.optim.Adam(self.model.sac_network.actor.parameters(),
                                                lr=float(self.config['actor_lr']),
                                                betas=self.config.get("actor_betas", [0.9, 0.999]))

        self.critic_optimizer = torch.optim.Adam(self.model.sac_network.critic.parameters(),
                                                 lr=float(self.config["critic_lr"]),
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=float(self.config["alpha_lr"]),
                                                    betas=self.config.get("alphas_betas", [0.9, 0.999]))

        self.replay_buffer = experience.VectorizedReplayBuffer(self.env_info['observation_space'].shape,
        self.env_info['action_space'].shape,
        self.replay_buffer_size,
        self._device)
        self.target_entropy_coef = config.get("target_entropy_coef", 1.0)
        self.target_entropy = self.target_entropy_coef * - self.env_info['action_space'].shape[0]
        print("Target entropy", self.target_entropy)

        self.algo_observer = config['features']['observer']

    # def load_networks(self, params):
    #     builder = model_builder.ModelBuilder()
    #     self.config['network'] = builder.load(params)

    def base_init(self, base_name, config):
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self._device = config.get('device', 'cuda:0')

        #temporary for Isaac gym compatibility
        print('Env info:')
        print(self.env_info)

        self.rewards_shaper = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        #self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.c_loss = nn.MSELoss()
        # self.c2_loss = nn.SmoothL1Loss()

        self.save_best_after = config.get('save_best_after', 500)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.save_freq = config.get('save_frequency', 0)

        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.obs_shape = self.observation_space.shape

        # self.games_to_track = self.config.get('games_to_track', 100)
        # self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        # self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.obs = None

        # self.min_alpha = torch.tensor(np.log(1)).float().to(self._device)

        self.frame = 0
        self.epoch_num = 0
        self.update_time = 0
        self.last_mean_rewards = -1000000000
        self.play_time = 0

        # TODO: put it into the separate class
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
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

        self.writer = SummaryWriter('runs/' + config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))
        print("Run Directory:", config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None
  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
