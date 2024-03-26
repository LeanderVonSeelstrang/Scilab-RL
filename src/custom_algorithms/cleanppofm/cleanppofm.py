import io
import pathlib
import warnings
from collections import deque
from typing import Dict, Optional, Tuple, Union
import math

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.nn import functional as F

from custom_algorithms.cleanppofm.forward_model import ProbabilisticSimpleForwardNet, \
    ProbabilisticForwardNetPositionPrediction
from utils.custom_buffer import CustomDictRolloutBuffer as DictRolloutBuffer
from utils.custom_buffer import CustomRolloutBuffer as RolloutBuffer
from utils.custom_wrappers import DisplayWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def flatten_obs(obs):
    if "agent" in obs and "target" in obs:
        agent, target = obs['agent'], obs['target']
        if isinstance(agent, np.ndarray):
            agent = torch.from_numpy(agent).to(device)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).to(device)
        return torch.cat([agent, target], dim=1).to(dtype=torch.float32).detach().clone()
    else:
        observation, ag, dg = obs["observation"], obs["achieved_goal"], obs["desired_goal"]
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).to(device)
        if isinstance(ag, np.ndarray):
            ag = torch.from_numpy(ag).to(device)
        if isinstance(dg, np.ndarray):
            dg = torch.from_numpy(dg).to(device)
        return torch.cat([observation, ag, dg], dim=1).to(dtype=torch.float32)


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        if isinstance(env.observation_space, spaces.Dict):
            obs_shape = np.sum([obs_space.shape for obs_space in env.observation_space.spaces.values()])
            self.flatten = True
        else:
            obs_shape = np.array(env.observation_space.shape).prod()
            self.flatten = False

        if isinstance(env.action_space, spaces.Discrete):
            action_shape = env.action_space.n
            self.discrete_actions = True
        else:
            action_shape = np.prod(env.action_space.shape)
            self.discrete_actions = False

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

        self.env = env

    def get_value(self, x):
        if self.flatten:
            x = flatten_obs(x)
        else:
            x = torch.tensor(x, device=device, dtype=torch.float32).detach().clone()
        return self.critic(x)

    def get_action_and_value(self, fm_network, x, action=None, deterministic=False, logger=None,
                             position_predicting=False):
        if self.flatten:
            x = flatten_obs(x)
        else:
            x = torch.tensor(x, device=device, dtype=torch.float32).detach().clone()

        if self.discrete_actions:
            action_mean = self.actor_mean(x)
            distribution = Categorical(logits=action_mean)
            if action is None:
                if deterministic:
                    action = torch.argmax(action_mean)
                    forward_normal_action = action.unsqueeze(0).unsqueeze(0)
                else:
                    action = distribution.sample()
                    forward_normal_action = action.unsqueeze(0)
            else:
                forward_normal_action = action.unsqueeze(1)
            # predict selected action
            # formal_normal_action in form of tensor([[action]])
            if position_predicting:
                positions = []
                for obs_element in x:
                    first_index_with_one = np.where(obs_element.cpu() == 1)[0][0] + 1
                    positions.append(first_index_with_one)
                positions = torch.tensor(positions, device=device).unsqueeze(1)
                forward_normal = fm_network(positions, forward_normal_action.float())
            else:
                forward_normal = fm_network(x, forward_normal_action.float())

            # TODO: put prediction of fm network into observation --> standard deviation or whole observation?
            # std describes the (un-)certainty of the prediction of each pixel
            # for index, element in enumerate(forward_normal.stddev[0]):
            #     logger.record(f"fm/stddev_{index}", element)
            # # loc describes the predicted position values
            # for index, element in enumerate(forward_normal.loc[0]):
            #     logger.record(f"fm/loc_{index}", element)
            logger.record_mean("fm/stddev", forward_normal.stddev.mean().item())
            return action.unsqueeze(0), distribution.log_prob(action), distribution.entropy(), self.critic(
                x), forward_normal
        else:
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            distribution = Normal(action_mean, action_std)
            if action is None:
                if deterministic:
                    action = action_mean
                else:
                    action = distribution.sample()
            return action, distribution.log_prob(action).sum(1), distribution.entropy().sum(1), self.critic(x)


class CLEANPPOFM:
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    This is a simplified one-file version of the stable-baselines3 PPO implementation.

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param env: The environment to learn from
    :param learning_rate: The learning rate
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter
    :param clip_range_vf: Clipping parameter for the value function
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    """

    def __init__(
            self,
            env: Union[GymEnv, str],
            learning_rate: float = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: Union[None, float] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            fm: dict = {},
            position_predicting: bool = False,
    ):
        self.num_timesteps = 0
        self.learning_rate = learning_rate
        self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        # Buffers for logging
        self.ep_info_buffer = None  # type: Optional[deque]

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = env.num_envs
        self.env = env

        if isinstance(self.action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([self.action_space.low, self.action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.fm = fm
        self.position_predicting = position_predicting
        if self.position_predicting:
            self.fm_network = ProbabilisticForwardNetPositionPrediction(self.env, self.fm).to(device)
        else:
            self.fm_network = ProbabilisticSimpleForwardNet(self.env, self.fm).to(device)
        self.fm_optimizer = torch.optim.Adam(
            self.fm_network.parameters(), lr=self.fm["learning_rate"]
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                    batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        # Check that `n_steps * n_envs > 1` to avoid NaN
        # when doing advantage normalization
        buffer_size = self.env.num_envs * self.n_steps
        assert buffer_size > 1 or (
            not normalize_advantage
        ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
        # Check that the rollout buffer size is a multiple of the mini-batch size
        untruncated_batches = buffer_size // batch_size
        if buffer_size % batch_size > 0:
            warnings.warn(
                f"You have specified a mini-batch size of {batch_size},"
                f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                f" after every {untruncated_batches} untruncated mini-batches,"
                f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
            )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.logger = None

        self._setup_model()

    def _setup_model(self) -> None:
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = Agent(self.env).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            ##### PROBLEM: rollout_buffer.get(self.batch_size) returns a random sample of the buffer,
            # it is not ordered anymore
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                observations = rollout_data.observations
                next_observations = rollout_data.next_observations

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                self.train_fm(observations, next_observations, actions)

                _, log_prob, entropy, values, _ = self.policy.get_action_and_value(fm_network=self.fm_network,
                                                                                   x=observations,
                                                                                   action=actions, logger=self.logger,
                                                                                   position_predicting=self.position_predicting)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        var_y = np.var(self.rollout_buffer.values.flatten())
        explained_var = np.nan if var_y == 0 else (
                1 - np.var(self.rollout_buffer.returns.flatten() - self.rollout_buffer.values.flatten()) / var_y)

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1
    ):
        iteration = 0
        self._last_obs = self.env.reset()
        callback.init_callback(self)
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer)

            if continue_training is False:
                break

            iteration += 1

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :return: True if function returned with at least `self.n_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        n_steps = 0
        elements_in_rollout_buffer = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while elements_in_rollout_buffer < self.n_steps:
            with torch.no_grad():
                actions, log_probs, _, values, forward_normal = self.policy.get_action_and_value(
                    fm_network=self.fm_network,
                    x=self._last_obs, logger=self.logger, position_predicting=self.position_predicting)
            # FIXME: very ugly coding
            #  when display wrapper is included, one "env" more is needed
            if isinstance(self.env.envs[0], DisplayWrapper):
                self.env.envs[0].env.env.env.env.forward_model_prediction = forward_normal.mean.cpu()
                self.env.envs[0].env.env.env.env.forward_model_stddev = forward_normal.stddev.cpu()
            else:
                self.env.envs[0].env.env.env.forward_model_prediction = forward_normal.mean.cpu()
                self.env.envs[0].env.env.env.forward_model_stddev = forward_normal.stddev.cpu()
            actions = actions.cpu().numpy()
            log_prob_float = float(np.mean(log_probs.cpu().numpy()))
            self.logger.record("train/rollout_logprob_step", float(log_prob_float))
            self.logger.record_mean("train/rollout_logprob_mean", float(log_prob_float))
            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            elif isinstance(self.action_space, spaces.Discrete):
                clipped_actions = actions[0]

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # reduce reward when prediction is bad
            flatten_last_obs = self._last_obs
            flatten_new_obs = new_obs
            if isinstance(env.observation_space, spaces.Dict):
                flatten_last_obs = flatten_obs(flatten_last_obs)
                flatten_new_obs = flatten_obs(flatten_new_obs)
            if not self.position_predicting:
                forward_normal = self.fm_network(flatten_last_obs, torch.from_numpy(actions))
            else:
                # get position out of observation
                # FIXME: this is hardcoded for the moonlander env
                positions = []
                for obs_element in new_obs:
                    first_index_with_one = np.where(obs_element == 1)[0][0] + 1
                    positions.append(first_index_with_one)
                # dtype = torch.float32 because action above (torch.from_numpy(actions)) is this type
                positions = torch.tensor(positions, device=device, dtype=torch.float32).unsqueeze(1)
                forward_normal = self.fm_network(positions, torch.from_numpy(actions))

            self.logger.record("train/rewards_without_stddev", rewards)
            # THIS DOESN'T WORK BECAUSE THE STDDEV IS WAY TOO LOW BECAUSE THE FM IS TRAINED WITHOUT INPUT NOISE
            # rewards -= forward_normal.stddev.mean().item() * 10
            # calculate manually prediction error (Euclidean distance) --> done in environment!
            # rewards -= math.sqrt(torch.sum((forward_normal.mean - flatten_new_obs) ** 2)) * 10

            self.logger.record("train/rollout_rewards_step", float(rewards.mean()))
            self.logger.record_mean("train/rollout_rewards_mean", float(rewards.mean()))
            # this is only logged when no hyperparameter tuning is running?
            # dodge/collect env
            if "simple" in infos[0].keys():
                self.logger.record("rollout_reward_simple", float(infos[0]["simple"]))
                self.logger.record("rollout_reward_gaussian", float(infos[0]["gaussian"]))
                self.logger.record("rollout_reward_pos_neg",
                                   float(infos[0]["pos_neg"]["pos"][0] + infos[0]["pos_neg"]["neg"][0]))
                self.logger.record("rollout_number_of_crashed_or_collected_objects",
                                   float(infos[0]["number_of_crashed_or_collected_objects"]))
            # gridworld env
            if "self.input_noise_is_applied_in_this_episode" in infos[0].keys():
                self.logger.record("input_noise_applied", infos[0]["self.input_noise_is_applied_in_this_episode"])
            # meta env
            if "dodge" in infos[0].keys():
                self.logger.record("dodge_gaussian_reward", infos[0]["dodge"]["gaussian"])
                self.logger.record("collect_gaussian_reward", infos[0]["collect"]["gaussian"])
                self.logger.record("task_switching_costs", infos[0]["task_switching_costs"])
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = infos[idx]["terminal_observation"]
                    with torch.no_grad():
                        terminal_obs["agent"] = np.expand_dims(terminal_obs["agent"], axis=0)
                        terminal_obs["target"] = np.expand_dims(terminal_obs["target"], axis=0)
                        terminal_value = self.policy.get_value(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # this is needed because otherwise the last observation and action does not match the new observation
            if not infos[0]["TimeLimit.truncated"]:
                rollout_buffer.add(
                    self._last_obs,
                    new_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                )
                elements_in_rollout_buffer += 1
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.get_value(new_obs)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train_fm(self, observations, next_observations, actions):
        if not self.position_predicting:
            if self.policy.flatten:
                observations = flatten_obs(observations)
                next_observations = flatten_obs(next_observations)
            ##### WHILE THE AGENT IS TRAINED WITH INPUT NOISE, THE FM IS TRAINED WITHOUT INPUT NOISE
            action_to_direction = {
                0: np.array([1, 0]),  # right
                1: np.array([1, 1]),  # right down (diagonal)
                2: np.array([0, 1]),  # down
                3: np.array([-1, 1]),  # left down (diagonal)
                4: np.array([-1, 0]),  # left
                5: np.array([-1, -1]),  # left up
                6: np.array([0, -1]),  # up
                7: np.array([1, -1])  # right up
            }
            agent_location_without_input_noise = torch.empty(size=(observations.shape[0], 4), device=device)
            for index, action in enumerate(actions):
                direction = action_to_direction[int(action)]
                # We use `np.clip` to make sure we don't leave the grid
                standard_agent_location = np.clip(
                    np.array(observations[index][0:2]) + direction, 0, 4
                )
                agent_location_without_input_noise[index] = torch.tensor(
                    np.concatenate((standard_agent_location, observations[index][2:4])), device=device
                )
            forward_normal = self.fm_network(observations, actions.float().unsqueeze(1))
            # log probs is the logarithm of the maximum likelihood
            # log because the deviation is easier (addition instead of multiplication)
            # negative because likelihood normally maximizes
            fw_loss = -forward_normal.log_prob(agent_location_without_input_noise)
        else:
            # get position out of observation
            # FIXME: this is hardcoded for the moonlander env
            positions = []
            for obs_element in observations:
                first_index_with_one = np.where(obs_element.cpu() == 1)[0][0] + 1
                positions.append(first_index_with_one)
            positions = torch.tensor(positions, device=device).unsqueeze(1)

            next_positions = []
            for next_obs_element in next_observations:
                first_index_with_one = np.where(next_obs_element.cpu() == 1)[0][0] + 1
                next_positions.append(first_index_with_one)
            next_positions = torch.tensor(next_positions, device=device).unsqueeze(1)

            forward_normal = self.fm_network(positions, actions.float().unsqueeze(1))
            # log probs is the logarithm of the maximum likelihood
            # log because the deviation is easier (addition instead of multiplication)
            # negative because likelihood normally maximizes
            fw_loss = -forward_normal.log_prob(next_positions)
        loss = fw_loss.mean()

        self.logger.record("fm/fw_loss", loss.item())
        self.fm_optimizer.zero_grad()
        loss.backward()
        self.fm_optimizer.step()

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        with torch.no_grad():
            action, _, _, _, forward_normal = self.policy.get_action_and_value(fm_network=self.fm_network,
                                                                               x=observation,
                                                                               deterministic=deterministic,
                                                                               logger=self.logger,
                                                                               position_predicting=self.position_predicting)
        return action.cpu().numpy(), state, forward_normal

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        # Copy parameter list, so we don't mutate the original dict
        data = self.__dict__.copy()
        for to_exclude in ["logger", "env", "num_timesteps", "policy",
                           "_last_obs", "_last_episode_starts"]:
            del data[to_exclude]
        # save network parameters
        data["_policy"] = self.policy.state_dict()
        data["_fm"] = self.fm_network.state_dict()
        torch.save(data, path)

    @classmethod
    def load(cls, path, env, **kwargs):
        model = cls(env=env, **kwargs)
        loaded_dict = torch.load(path, map_location=torch.device(device))
        for k in loaded_dict:
            if k not in ["_policy", "_fm"]:
                model.__dict__[k] = loaded_dict[k]
        # load network states
        model.policy.load_state_dict(loaded_dict["_policy"])
        model.fm_network.load_state_dict(loaded_dict["_fm"])
        return model

    def set_logger(self, logger):
        self.logger = logger

    def get_env(self):
        return self.env
