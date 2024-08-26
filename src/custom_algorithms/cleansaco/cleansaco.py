from typing import Dict, Optional, Tuple, Union
from copy import deepcopy

import pathlib
import io
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gymnasium import spaces
from torch.distributions import Normal

from stable_baselines3.common.logger import Logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, env, action_scale_factor=1.0):
        self.action_scale_factor = action_scale_factor
        super().__init__()
        obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
        self.fc1 = nn.Linear(obs_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0 * self.action_scale_factor,
                                    dtype=torch.float32)
        action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)

        self.register_buffer(
            "action_scale", action_scale
        )
        self.register_buffer(
            "action_bias", action_bias
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def get_action(self, x, deterministic=False):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
        self.fc1 = nn.Linear(obs_shape + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CriticEnsemble(nn.Module):
    def __init__(self, env, n_critics: int):
        super().__init__()
        self._critics = nn.ModuleList(
            [
                Critic(env)
                for _ in range(n_critics)
            ]
        )

    def forward(self, x, a):
        return torch.stack([critic(x, a) for critic in self._critics])


def flatten_obs(obs, device):
    observation, ag, dg = obs['observation'], obs['achieved_goal'], obs['desired_goal']
    if isinstance(observation, np.ndarray):
        observation = torch.from_numpy(observation).to(device)
    if isinstance(ag, np.ndarray):
        ag = torch.from_numpy(ag).to(device)
    if isinstance(dg, np.ndarray):
        dg = torch.from_numpy(dg).to(device)
    return torch.cat([observation, ag, dg], dim=1).to(dtype=torch.float32)


def kl_divergence(p, q):
    if not isinstance(p, torch.distributions.Normal) or not isinstance(q, torch.distributions.Normal):
        raise ValueError("Inputs not torch.distributions.Normal")
    kl_div = torch.distributions.kl.kl_divergence(p, q)
    kl_div = torch.mean(kl_div)
    kl_div_value = kl_div.item()
    return kl_div_value


class MCNORM:
    def __init__(self, history_length=400000, max_percentile=94, min_percentile=1):
        self.rewards = []
        self.max_percentile = max_percentile
        self.min_percentile = min_percentile
        self.history_length = history_length

    def update(self, value):
        if torch.is_tensor(value):
            value = value.cpu().item()
        self.rewards.append(value)

        if len(self.rewards) > self.history_length:
            self.rewards.pop(0)

    def normalize(self, value):
        if len(self.rewards) == 0:
            return value

        median = np.median(self.rewards)
        std_dev = np.std(self.rewards)

        if std_dev == 0:
            std_dev = 1e-8

        if torch.is_tensor(value):
            value = value.cpu().item()
        normalized_value = (value - median) / std_dev

        if np.isnan(normalized_value):
            normalized_value = 0.0

        # min_norm = (np.min(self.rewards) - median) / std_dev
        min_norm = (np.percentile(self.rewards, self.min_percentile) - median) / std_dev  # for symmetry 6, better set it to 0
        max_norm = (np.percentile(self.rewards, self.max_percentile) - median) / std_dev  # best empirical fit 94

        # Avoid potential NaNs
        if np.isnan(min_norm) or np.isnan(max_norm) or min_norm == max_norm:
            return 0

        if normalized_value > max_norm:
            return 1
        if normalized_value < min_norm:
            return 0

        # Scale and shift to the range [-1, 0]
        scaled_value = (normalized_value - min_norm) * (1 / (max_norm - min_norm))
        # Adapt the range of the MC reward to the studied task (e.g. FetchPush [-1, 0], AntMazeDense [0, 1])
        # Also change the conditions accordingly

        if np.isnan(scaled_value):
            scaled_value = 0 # -1.0

        return scaled_value


class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=5):
        super(TransitionModel, self).__init__()
        layers = []
        input_dim = state_dim + action_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dim, state_dim)
        self.log_std_layer = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.hidden_layers(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        return Normal(mean, std)


class ActionOnlyModel(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim=256, num_layers=5):
        super(ActionOnlyModel, self).__init__()
        layers = []
        input_dim = action_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dim, state_dim)
        self.log_std_layer = nn.Linear(hidden_dim, state_dim)

    def forward(self, action):
        x = self.hidden_layers(action)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        return Normal(mean, std)


class CLEANSACO:
    """
    A one-file version of SAC derived from both the CleanRL and stable-baselines3 versions of SAC.
    :param env: The Gym environment to learn from
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically
    :param use_her: whether to use hindsight experience replay (HER) by using the SB3 HerReplayBuffer
    """

    def __init__(
            self,
            env: GymEnv,
            learning_rate: float = 3e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 1000,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            ent_coef: Union[str, float] = "auto",
            use_her: bool = True,
            n_critics: int = 2,
            ignore_dones_for_qvalue: bool = False,
            action_scale_factor: float = 1.0,
            log_obs_step: bool = False,
            log_act_step: bool = False,
            mc_alpha: float = 0.5,
            max_percentile: float = 94,
            min_percentile: float = 1,
            mc_lr=0.0003
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.n_critics = n_critics
        self.ignore_dones_for_qvalue = ignore_dones_for_qvalue
        self.action_scale_factor = action_scale_factor
        self.log_obs_step = log_obs_step
        self.log_act_step = log_act_step
        self.max_percentile = max_percentile
        self.min_percentile = min_percentile
        self.mc_lr = mc_lr
        self.mc_alpha = mc_alpha  # Weight for the MC reward
        self.mc_normalizer = MCNORM(max_percentile=self.max_percentile,
                                    min_percentile=self.min_percentile)

        self.env = env
        if isinstance(self.env.action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([self.env.action_space.low, self.env.action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        obs_shape = np.sum([obs_space.shape for obs_space in self.env.observation_space.spaces.values()])
        action_dim = np.prod(self.env.action_space.shape)

        self.transition_model = TransitionModel(obs_shape, action_dim).to(self.device)
        self.action_only_model = ActionOnlyModel(action_dim, obs_shape).to(self.device)
        self.transition_optimizer = torch.optim.Adam(
            list(self.transition_model.parameters()) + list(self.action_only_model.parameters()),
            lr=self.mc_lr
        )

        # initialize replay buffer
        if use_her:
            self.replay_buffer = HerReplayBuffer(
                self.buffer_size,
                self.env.observation_space,
                self.env.action_space,
                env=self.env,
                device=self.device,
                n_envs=self.env.num_envs
            )
        else:
            self.replay_buffer = DictReplayBuffer(
                self.buffer_size,
                self.env.observation_space,
                self.env.action_space,
                device=self.device,
                n_envs=self.env.num_envs
            )

        self._create_actor_critic()

        self.ent_coef = ent_coef
        if self.ent_coef == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
            self.log_ent_coef = torch.zeros(1, device=self.device).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
        else:
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef), device=self.device)

        self.logger = None
        self._last_obs = None
        self.num_timesteps = 0
        self.episode_steps = 0
        self._n_updates = 0

    def _create_actor_critic(self) -> None:
        self.actor = Actor(self.env, self.action_scale_factor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic = CriticEnsemble(self.env, self.n_critics).to(self.device)
        self.critic_target = CriticEnsemble(self.env, self.n_critics).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval=None,
    ):
        callback.init_callback(self)
        callback.on_training_start(locals(), globals())

        self._last_obs = self.env.reset()
        self.episode_steps = 0
        while self.num_timesteps < total_timesteps:
            continue_training = self.step_env(callback=callback)

            if continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train()

        callback.on_training_end()

        return self

    def step_env(
            self,
            callback: BaseCallback
    ):
        """
        Take one step in the environment and store the transition in a ``ReplayBuffer``.
        In the stable-baselines3 version, this function is called "collect_rollouts" and it is possible to take
        multiple steps in the environment, but here we hard-code it to one step for simplicity.

        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :return: True if the training should continue, else False
        """
        # Select action randomly or according to policy
        if self.num_timesteps < self.learning_starts:
            action = np.array([self.env.action_space.sample()])
            log_pi = 0.0
        else:
            action, log_pi = self.predict(self._last_obs)
            log_pi = float(log_pi.mean())
        flat_obs = flatten_obs(self._last_obs, self.device)
        torch_obs = torch.tensor(flat_obs)
        torch_action = torch.tensor(action, device=self.device)
        q_val = float(self.critic(torch_obs, torch_action).mean())
        if self.ent_coef == "auto":
            ent_coef = self.log_ent_coef.exp().item()
        else:
            ent_coef = self.ent_coef_tensor
        ent_coef = float(ent_coef)
        self.logger.record_mean("train/rollout_ent_coef", ent_coef)
        self.logger.record("train/rollout_logpi_times_coef_step", log_pi * ent_coef)
        self.logger.record_mean("train/rollout_logpi_times_coef_mean", log_pi * ent_coef)
        self.logger.record("train/rollout_logpi_step", log_pi)
        self.logger.record_mean("train/rollout_logpi_mean", log_pi)
        self.logger.record("train/rollout_q_step", q_val)
        self.logger.record_mean("train/rollout_q_mean", q_val)
        # append mc and kld

        # perform action
        new_obs, rewards, dones, infos = self.env.step(action)
        self.episode_steps += 1
        self.logger.record("train/rollout_rewards_step", np.mean(rewards))
        self.logger.record_mean("train/rollout_rewards_mean", np.mean(rewards))
        if self.log_obs_step:
            for n in range(new_obs['observation'].shape[1]):
                dim_obs = new_obs['observation'][:, n]
                self.logger.record(f"train/obs_{n}", np.mean(dim_obs))
        if self.log_act_step:
            for n in range(action.shape[1]):
                dim_act = action[:, n]
                self.logger.record(f"train/act_{n}", np.mean(dim_act))

        self.num_timesteps += self.env.num_envs

        # save data to replay buffer; handle `terminal_observation`
        next_obs = deepcopy(new_obs)
        for i, done in enumerate(dones):
            if done:
                self.logger.record_mean(f'train/mean_ep_length', self.episode_steps)
                self.episode_steps = 0
                next_obs_ = infos[i]["terminal_observation"]
                for key in next_obs.keys():
                    next_obs[key][i] = next_obs_[key]
        self.replay_buffer.add(self._last_obs, next_obs, action, rewards, dones, infos)

        self._last_obs = new_obs

        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return False
        return True

    def train(self):
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        # Sample replay buffer
        replay_data = self.replay_buffer.sample(self.batch_size)
        observations = flatten_obs(replay_data.observations, self.device)
        next_observations = flatten_obs(replay_data.next_observations, self.device)

        # Calculate morphological computation reward
        with torch.no_grad():
            predicted_next_states = self.transition_model(observations, replay_data.actions)
            predicted_next_states_no_state = self.action_only_model(replay_data.actions)

        mc_rewards = kl_divergence(predicted_next_states, predicted_next_states_no_state)
        self.logger.record("train/kld", mc_rewards)
        self.logger.record("train/max_percentile", self.max_percentile)
        self.logger.record("train/min_percentile", self.min_percentile)
        self.logger.record("train/mc_lr", self.mc_lr)
        self.mc_normalizer.update(mc_rewards)

        mc_rewards = self.mc_normalizer.normalize(mc_rewards)
        self.logger.record("train/mc_rewards", mc_rewards)
        alpha, beta = self.mc_alpha, 1 - self.mc_alpha

        combined_rewards = beta * replay_data.rewards.flatten() + alpha * mc_rewards
        self.logger.record("train/mc_alpha", alpha)
        self.logger.record("train/mc_beta", beta)

        # Optimize entropy coefficient
        if self.ent_coef == "auto":
            with torch.no_grad():
                _, log_pi = self.actor.get_action(observations)
            ent_coef_loss = (-self.log_ent_coef * (log_pi + self.target_entropy)).mean()
            self.logger.record("train/ent_coef_loss", ent_coef_loss.item())

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            ent_coef = self.log_ent_coef.exp().item()
        else:
            ent_coef = self.ent_coef_tensor
        self.logger.record("train/ent_coef", ent_coef)

        # Train critic
        with torch.no_grad():
            next_state_actions, next_state_log_pi = self.actor.get_action(next_observations)
            crit_next_targets = self.critic_target(next_observations, next_state_actions)
            min_crit_next_target = torch.min(crit_next_targets, dim=0).values
            min_crit_next_target -= ent_coef * next_state_log_pi
            if self.ignore_dones_for_qvalue:
                next_q_value = combined_rewards + self.gamma * min_crit_next_target.flatten()
            else:
                next_q_value = combined_rewards + (
                        1 - replay_data.dones.flatten()) * self.gamma * min_crit_next_target.flatten()

        critic_a_values = self.critic(observations, replay_data.actions)
        crit_loss = torch.stack([F.mse_loss(_a_v, next_q_value.view(-1, 1)) for _a_v in critic_a_values]).sum()
        self.logger.record("train/critic_loss", crit_loss.item())
        self.logger.record("train/train_rewards", replay_data.rewards.flatten().mean().item())

        self.critic_optimizer.zero_grad()
        crit_loss.backward()
        self.critic_optimizer.step()

        # Train actor
        pi, log_pi = self.actor.get_action(observations)
        min_crit_pi = torch.min(self.critic(observations, pi), dim=0).values
        actor_loss = ((ent_coef * log_pi) - min_crit_pi).mean()
        self.logger.record("train/actor_loss", actor_loss.item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks with polyak update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.mul_(1 - self.tau)
            torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

        predicted_next_state_dist = self.transition_model(observations, replay_data.actions)
        predicted_next_state_no_state_dist = self.action_only_model(replay_data.actions)

        # Compute MC loss
        transition_loss = -predicted_next_state_dist.log_prob(next_observations).mean() \
                          - predicted_next_state_no_state_dist.log_prob(next_observations).mean()

        # Logging and optimization
        self.logger.record("train/transition_loss", transition_loss.item())
        self.transition_optimizer.zero_grad()
        transition_loss.backward()
        self.transition_optimizer.step()

    def predict(
            self,
            obs: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """
        Get the policy action given an observation.

        :param obs: the input observation
        :return: the model's action
        """
        observation = flatten_obs(obs, self.device)
        action, log_pi = self.actor.get_action(observation, deterministic=deterministic)
        return action.detach().cpu().numpy(), log_pi.detach().cpu().numpy()

    def save(self, path: Union[str, pathlib.Path, io.BufferedIOBase]):
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in ["logger", "env", "num_timesteps", "_n_updates", "_last_obs",
                           "replay_buffer", "actor", "critic", "critic_target"]:
            del data[to_exclude]
        # save network parameters
        data["_actor"] = self.actor.state_dict()
        data["_critic"] = self.critic.state_dict()
        torch.save(data, path)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        loaded_dict = torch.load(path)
        for k in loaded_dict:
            if k not in ["_actor", "_critic"]:
                model.__dict__[k] = loaded_dict[k]
        # load network states
        model.actor.load_state_dict(loaded_dict["_actor"])
        model.critic.load_state_dict(loaded_dict["_critic"])
        return model

    def set_logger(self, logger: Logger) -> None:
        self.logger = logger

    def get_env(self):
        return self.env
