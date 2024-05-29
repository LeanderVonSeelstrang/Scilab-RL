import copy
import os
import sys

import gymnasium as gym
import numpy as np

# FIXME: needed for rendering rgb array
np.set_printoptions(threshold=sys.maxsize)
import yaml
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from custom_algorithms.cleanppo.cleanppo import CLEANPPO
from stable_baselines3.common.env_util import make_vec_env
from custom_envs.moonlander.moonlander_env import MoonlanderWorldEnv


class MetaEnvPretrained(gym.Env):
    render_mode = None
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, reward_function: str = "gaussian"):
        self.ROOT_DIR = "."
        config_path_dodge_asteroids = os.path.join(os.path.dirname(os.path.realpath(__file__)), "standard_config.yaml")
        config_path_collect_asteroids = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     "standard_config_second_task.yaml")

        with open(config_path_dodge_asteroids, "r") as file:
            config_dodge_asteroids = yaml.safe_load(file)
        with open(config_path_collect_asteroids, "r") as file:
            config_collect_asteroids = yaml.safe_load(file)

        # FIXME: missing tests that configurations are the same in both files
        agent_config = config_dodge_asteroids["agent"]
        world_config = config_dodge_asteroids["world"]

        ### ACTION SPACE ###
        # one action to decide which task to control
        # action 0 --> task 0 can be controlled
        # action 1 --> task 1 can be controlled
        self.action_space = gym.spaces.Discrete(2)

        ### OBSERVATION SPACE ###
        # 10x12 grid for each task
        # TODO: in the future --> SoC for each task
        self.y_position_of_agent = agent_config["size"]
        self.following_observations_size = min(
            agent_config["observation_height"],
            int(world_config["y_height"] - self.y_position_of_agent + 1),
        )
        self.observation_space = gym.spaces.Box(
            low=-10,
            high=5,
            shape=(
                self.following_observations_size * (world_config["x_width"] + 2) * 2,
            ),
            dtype=np.int64,
        )

        ### REWARD RANGE ###
        # FIXME: not sure about this
        # reward is added from each task

        # first state
        self.dodge_asteroids = MoonlanderWorldEnv(task="dodge", reward_function=reward_function)
        self.collect_asteroids = MoonlanderWorldEnv(task="collect", reward_function=reward_function)
        self.state_of_dodge_asteroids, _ = self.dodge_asteroids.reset()
        self.state_of_collect_asteroids, _ = self.collect_asteroids.reset()

        # Load the trained agents
        # FIXME: this is an ugly hack to load the trained agents
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../../../policies/dodge_small_rl_model_best"), "rb"
        ) as file:
            print("start loading agents", file)
            self.trained_dodge_asteroids = CLEANPPO.load(path=file,
                                                         env=make_vec_env("MoonlanderWorld-dodge-gaussian-v0",
                                                                          n_envs=1))
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../../../policies/collect_small_rl_model_best"), "rb"
        ) as file:
            # same model cannot be loaded twice -> copy does also not work
            self.trained_collect_asteroids = CLEANPPO.load(path=file,
                                                           env=make_vec_env("MoonlanderWorld-collect-gaussian-v0",
                                                                            n_envs=1))
            print("finish loading agents")

        # concatenate state with vector of zeros
        self.mask = np.full(shape=self.state_of_collect_asteroids.shape, fill_value=5)
        self.current_task = 0
        self.state = np.concatenate(
            (self.state_of_dodge_asteroids.reshape(10, 12), self.mask.reshape(10, 12)),
            axis=1,
        ).flatten()

        # for rendering
        plt.ion()
        self.fig, self.ax = plt.subplots()
        eximg = np.zeros((10, 24))
        eximg[0] = -10
        eximg[1] = 5
        self.im = self.ax.imshow(eximg)

        # counter
        self.episode_counter = 0
        self.step_counter = 0
        self.switch_counter = 0

    def step(self, action: int):
        """
        action: selects the task
                0: one
                1: two
        """
        task_switching_costs = 0
        # action 0,1, or 2 (left, stay, right) for each task
        match action:

            case 0:
                # dodge task
                active_model = self.trained_dodge_asteroids
                active_env = self.dodge_asteroids
                inactive_env = self.collect_asteroids
                last_state = self.state_of_dodge_asteroids
                self.current_task = 0
            case 1:
                # collect task
                active_model = self.trained_collect_asteroids
                active_env = self.collect_asteroids
                inactive_env = self.dodge_asteroids
                last_state = self.state_of_collect_asteroids
                self.current_task = 1

            # If an exact match is not confirmed, this last case will be used if provided
            case _:
                raise ValueError("action must be 0, 1, 2, or 3")

        # predict next action
        action_of_task_agent, _ = active_model.predict(np.expand_dims(last_state, 0), deterministic=True)
        # perform action
        new_state, active_reward, active_is_done, _, active_info = active_env.step(
            action=action_of_task_agent[0])

        # perform action in inactive task
        _, inactive_reward, inactive_is_done, _, inactive_info = inactive_env.step(action=0)

        # update last state
        # one only has the last state of the last actively acting TODO: prediction of FM?
        match action:
            case 0:
                # dodge task
                self.state_of_dodge_asteroids = new_state
            case 1:
                # collect task
                self.state_of_collect_asteroids = new_state
        if self.current_task == 0:
            self.state = np.concatenate(
                (
                    self.state_of_dodge_asteroids.reshape(10, 12),
                    self.mask.reshape(10, 12),
                ),
                axis=1,
            ).flatten()
        elif self.current_task == 1:
            self.state = np.concatenate(
                (
                    self.mask.reshape(10, 12),
                    self.state_of_collect_asteroids.reshape(10, 12),
                ),
                axis=1,
            ).flatten()

        self.step_counter += 1
        if action == 0:
            info_dodge = active_info
            reward_dodge = active_reward
            info_collect = inactive_info
            reward_collect = inactive_reward
        else:
            info_dodge = inactive_info
            reward_dodge = inactive_reward
            info_collect = active_info
            reward_collect = active_reward
        info = {"info_dodge": info_dodge, "info_collect": info_collect, "task_switching_costs": task_switching_costs,
                "reward_dodge": reward_dodge, "reward_collect": reward_collect}
        return (
            self.state,
            active_reward + inactive_reward,
            active_is_done or inactive_is_done,
            False,
            info,
        )

    def render(self):
        observation = copy.deepcopy(self.state).reshape(10, 24)
        self.im.set_data(observation)
        if self.render_mode == "human":
            self.fig.canvas.draw_idle()
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))

    def reset(self, seed=None, options=None):
        self.state_of_dodge_asteroids, _ = self.dodge_asteroids.reset()
        self.state_of_collect_asteroids, _ = self.collect_asteroids.reset()
        # concatenate state with vector of zeros
        self.mask = np.full(shape=self.state_of_collect_asteroids.shape, fill_value=5)
        self.current_task = 0
        self.state = np.concatenate(
            (self.state_of_dodge_asteroids.reshape(10, 12), self.mask.reshape(10, 12)),
            axis=1,
        ).flatten()

        # counter
        self.episode_counter += 1
        self.step_counter = 0
        return self.state, {}
