import copy
import os
import sys
import torch
import gymnasium as gym
import numpy as np
import yaml
from matplotlib import pyplot as plt
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

# FIXME: needed for rendering rgb array
np.set_printoptions(threshold=sys.maxsize)

from custom_algorithms.cleanppofm.cleanppofm import CLEANPPOFM
from custom_algorithms.cleanppofm.utils import reward_estimation, get_position_and_object_positions_of_observation, \
    get_observation_of_position_and_object_positions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # test if configurations are the same
        config_dodge_asteroids_copy = config_dodge_asteroids.copy()
        config_dodge_asteroids_copy["world"]["objects"].pop("type")
        config_collect_asteroids_copy = config_collect_asteroids.copy()
        config_collect_asteroids_copy["world"]["objects"].pop("type")

        if not config_dodge_asteroids_copy == config_collect_asteroids_copy:
            raise ValueError("Configurations are not the same")

        agent_config = config_dodge_asteroids["agent"]
        world_config = config_dodge_asteroids["world"]

        ### ACTION SPACE ###
        # one action to decide which task to control
        # action 0 --> task 0 can be controlled
        # action 1 --> task 1 can be controlled
        self.action_space = gym.spaces.Discrete(2)

        ### OBSERVATION SPACE ###
        # 10x12 grid for each task
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

        # logger
        tmp_path = "/tmp/sb3_log/"
        self.logger = configure(tmp_path, ["stdout", "csv"])

        # Load the trained agents
        # FIXME: this is an ugly hack to load the trained agents
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../../../policies/dodge_object_prediction_rl_model_best"), "rb"
        ) as file:
            print("start loading agents", file)
            self.trained_dodge_asteroids = CLEANPPOFM.load(path=file,
                                                           env=make_vec_env("MoonlanderWorld-dodge-gaussian-v0",
                                                                            n_envs=1))
            self.trained_dodge_asteroids.set_logger(logger=self.logger)
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../../../policies/collect_object_prediction_rl_model_best"), "rb"
        ) as file:
            # same model cannot be loaded twice -> copy does also not work
            self.trained_collect_asteroids = CLEANPPOFM.load(path=file,
                                                             env=make_vec_env("MoonlanderWorld-collect-gaussian-v0",
                                                                              n_envs=1))
            self.trained_collect_asteroids.set_logger(logger=self.logger)
            print("finish loading agents")

        # test if models used the same configuration during training
        if not (
                self.trained_dodge_asteroids.position_predicting == self.trained_collect_asteroids.position_predicting
                and self.trained_dodge_asteroids.reward_predicting == self.trained_collect_asteroids.reward_predicting
                and self.trained_dodge_asteroids.number_of_future_steps == self.trained_collect_asteroids.number_of_future_steps
                and self.trained_dodge_asteroids.fm_trained_with_input_noise == self.trained_collect_asteroids.fm_trained_with_input_noise
                and self.trained_dodge_asteroids.input_noise_on == self.trained_collect_asteroids.input_noise_on
                and self.trained_dodge_asteroids.maximum_number_of_objects == self.trained_collect_asteroids.maximum_number_of_objects):
            raise ValueError("Models used different configurations during training")

        # the state could possibly be a belief state of the forward model
        # only one return value because DummyVecEnv only returns one observation
        self.state_of_dodge_asteroids = self.trained_dodge_asteroids.env.reset()
        self.state_of_collect_asteroids = self.trained_collect_asteroids.env.reset()

        # concatenate states (possibly belief states)
        self.current_task = 0
        self.state = np.concatenate(
            (self.state_of_dodge_asteroids.reshape(10, 12), self.state_of_collect_asteroids.reshape(10, 12)),
            axis=1,
        ).flatten()

        self.SoC_dodge = 0
        self.SoC_collect = 0

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
        # action 0: dodge asteroids
        # action 1: collect asteroids
        match action:

            case 0:
                # dodge task
                active_model = self.trained_dodge_asteroids
                active_last_state = self.state_of_dodge_asteroids
                inactive_model = self.trained_collect_asteroids
                inactive_SoC = self.SoC_collect
                inactive_last_state = self.state_of_collect_asteroids
                self.current_task = 0
            case 1:
                # collect task
                active_model = self.trained_collect_asteroids
                active_last_state = self.state_of_collect_asteroids
                inactive_model = self.trained_dodge_asteroids
                inactive_SoC = self.SoC_dodge
                inactive_last_state = self.state_of_dodge_asteroids
                self.current_task = 1

            # If an exact match is not confirmed, this last case will be used if provided
            case _:
                raise ValueError("action must be 0, 1")

        ### ACTIVE TASK ###
        # predict next action
        action_of_task_agent, _, _ = active_model.predict(active_last_state, deterministic=True)
        # get position and object positions of observation
        active_agent_and_object_positions_tensor = get_position_and_object_positions_of_observation(
            torch.tensor(active_last_state, device=device))
        # forward model predictions once with state and action
        active_belief_state_normal_distribution = active_model.fm_network(active_agent_and_object_positions_tensor,
                                                                          torch.tensor([action_of_task_agent]).float())
        # perform action & SoC calculation & reward estimation corrected by SoC
        new_state, _, active_is_done, active_info, active_prediction_error, active_reward_estimation_corrected_by_SoC = active_model.step_in_env(
            actions=torch.tensor(action_of_task_agent).float(),
            forward_normal=active_belief_state_normal_distribution)
        active_SoC = 1 - active_prediction_error

        ### INACTIVE TASK ###
        # perform default action 0 in inactive task
        # only four return value because DummyVecEnv only returns observation, reward, done, info
        _, _, inactive_is_done, inactive_info = inactive_model.env.step(torch.tensor([0], device=device))
        # get position and object positions of observation
        inactive_agent_and_object_positions_tensor = get_position_and_object_positions_of_observation(
            torch.tensor(inactive_last_state, device=device))
        # forward model predictions once with state and action
        inactive_belief_state_normal_distribution = inactive_model.fm_network(
            inactive_agent_and_object_positions_tensor,
            torch.tensor([[0]]).float())
        # SoC update --> degrade SoC by 0.1
        inactive_SoC = min(max(0, inactive_SoC - 0.1), 1)
        # reward estimation corrected by SoC
        inactive_reward_estimation_corrected_by_SoC = reward_estimation(fm_network=inactive_model.fm_network,
                                                                        new_obs=
                                                                        inactive_belief_state_normal_distribution.mean[
                                                                            0][:-1].unsqueeze(0),
                                                                        env_name="MoonlanderWorldEnv",
                                                                        rewards=
                                                                        inactive_belief_state_normal_distribution.mean[
                                                                            0][-1],
                                                                        prediction_error=(1 - inactive_SoC),
                                                                        # we already have the positions through the prediction
                                                                        position_predicting=False,
                                                                        number_of_future_steps=inactive_model.number_of_future_steps,
                                                                        maximum_number_of_objects=inactive_model.maximum_number_of_objects)
        inactive_reward_estimation_corrected_by_SoC = inactive_reward_estimation_corrected_by_SoC.item()

        belief_state = get_observation_of_position_and_object_positions(
            inactive_belief_state_normal_distribution.mean[0][:-1].cpu().unsqueeze(0)).flatten().cpu().numpy()
        belief_state = np.expand_dims(belief_state, 0)
        match action:
            case 0:
                # dodge task
                self.state_of_dodge_asteroids = new_state
                info_dodge = active_info
                reward_dodge = active_reward_estimation_corrected_by_SoC
                self.SoC_dodge = active_SoC
                self.state_of_collect_asteroids = belief_state
                info_collect = inactive_info
                reward_collect = inactive_reward_estimation_corrected_by_SoC
                self.SoC_collect = inactive_SoC
            case 1:
                # collect task
                self.state_of_dodge_asteroids = belief_state
                info_dodge = inactive_info
                reward_dodge = inactive_reward_estimation_corrected_by_SoC
                self.SoC_dodge = inactive_SoC
                self.state_of_collect_asteroids = new_state
                info_collect = active_info
                reward_collect = active_reward_estimation_corrected_by_SoC
                self.SoC_collect = active_SoC
            case _:
                raise ValueError("action must be 0, 1")

        self.state = np.concatenate(
            (
                self.state_of_dodge_asteroids.reshape(10, 12),
                self.state_of_collect_asteroids.reshape(10, 12),
            ),
            axis=1,
        ).flatten()

        self.step_counter += 1
        info = {"info_dodge": info_dodge, "info_collect": info_collect, "reward_dodge": reward_dodge,
                "reward_collect": reward_collect}
        return (
            self.state,
            active_reward_estimation_corrected_by_SoC + inactive_reward_estimation_corrected_by_SoC,
            active_is_done or inactive_is_done,
            False,
            info,
        )

    def render(self):
        observation = copy.deepcopy(self.state).reshape(10, 24)
        # place frame around current task
        if self.current_task == 0:
            first_fill_value = -10
            second_fill_value = -1
            row = np.array(
                [[-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -1, -1, -1, -1, -1, -1, -1, -1,
                  -1, -1,
                  -1, -1, -1, -1]])
        else:
            first_fill_value = -1
            second_fill_value = -10
            row = np.array(
                [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -10, -10, -10, -10, -10, -10, -10, -10,
                  -10, -10, -10, -10, -10, -10]])

        observation = np.concatenate(
            (
                np.full((10, 1), first_fill_value),
                observation[:, :12],
                np.full((10, 1), first_fill_value),
                np.full((10, 1), second_fill_value),
                observation[:, 12:],
                np.full((10, 1), second_fill_value)),
            axis=1)
        observation = np.concatenate((row, observation, row), axis=0)

        self.im.set_data(observation)
        if self.render_mode == "human":
            self.fig.canvas.draw_idle()
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))

    def reset(self, seed=None, options=None):
        # the state could possibly be a belief state of the forward model
        # only one return value because DummyVecEnv only returns one observation
        self.state_of_dodge_asteroids = self.trained_dodge_asteroids.env.reset()
        self.state_of_collect_asteroids = self.trained_collect_asteroids.env.reset()

        # concatenate states (possibly belief states)
        self.current_task = 0
        self.state = np.concatenate(
            (self.state_of_dodge_asteroids.reshape(10, 12), self.state_of_collect_asteroids.reshape(10, 12)),
            axis=1,
        ).flatten()

        # counter
        self.episode_counter += 1
        self.step_counter = 0
        return self.state, {}
