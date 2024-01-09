import copy
import csv
import datetime
import logging
import math
import os
import random as rnd
from typing import List, Dict

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw
from gymnasium import Env
from gymnasium import spaces
from hydra.utils import get_original_cwd
from matplotlib import pyplot as plt

# FIXME
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

import custom_envs.moonlander.helper_functions as hlp


class MoonlanderWorldEnv(Env):
    render_mode = None
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, task: str = "dodge", reward_function: str = "pos_neg"):
        """
        initialises the environment
        Args:
        """
        self.ROOT_DIR = "."
        if task == "dodge":
            config_path = os.path.join(
                get_original_cwd(),
                "src/custom_envs/moonlander/standard_config.yaml",
            )
            # config_path = os.path.join(
            #     "/home/annika/coding_projects/scilab-new/Scilab-RL/src",
            #     "custom_envs/moonlander/standard_config.yaml",
            # )
        elif task == "collect":
            config_path = os.path.join(
                get_original_cwd(),
                "src/custom_envs/moonlander/standard_config_second_task.yaml",
            )
            # config_path = os.path.join(
            #     "/home/annika/coding_projects/scilab-new/Scilab-RL/src",
            #     "custom_envs/moonlander/standard_config_second_task.yaml",
            # )
        else:
            raise ValueError("Task {} not implemented".format(task))

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        # overwrite current reward function
        config["reward_function"] = reward_function

        # FIXED VARIABLES
        if (
                config["agent"]["size"] < 1
                or config["world"]["y_height"] < 1
                or config["world"]["x_width"] < 1
                or config["agent"]["observation_height"] < 1
        ):
            raise ValueError(
                "Only numbers greater than zero are allowed. Please redefine the size, world height or "
                "agent observation height"
            )

        if (
                (not 0.0 <= config["world"]["drift"]["invisible_drift_probability"] <= 1.0)
                or (not 0.0 <= config["world"]["drift"]["fake_drift_probability"] <= 1.0)
                or (
                not config["world"]["drift"]["invisible_drift_probability"]
                    + config["world"]["drift"]["fake_drift_probability"]
                    <= 1.0
        )
        ):
            raise ValueError(
                "invisible_drift_probability and fake_drift_probability must be in the range of [0, 1] and"
                "sum to less than or exactly one!"
            )
        if config["world"]["objects"]["type"] not in ["obstacle", "coin"]:
            raise ValueError(
                "object_type must be either 'obstacle' or 'coin' but was {}".format(
                    config["world"]["objects"]["type"]
                )
            )

        if config["world"]["drift"]["drift_at_whole_level"] not in [
            "empty",
            "left",
            "right",
            "ranges",
        ]:
            raise ValueError(
                "drift_at_whole_level must be one of 'empty', 'left', 'right' or 'ranges', but was {}".format(
                    config["world"]["drift"]["drift_at_whole_level"]
                )
            )

        self.config = config
        self.reward_function = config["reward_function"]
        self.already_crashed_objects = []
        if self.reward_function not in ["simple", "gaussian", "pos_neg"]:
            raise ValueError(
                "Reward function {} not implemented".format(self.reward_function)
            )
        self.pos_neg_reward_info_dict_per_step = {}
        self.gaussian_reward_info_per_step = 0
        self.simple_reward_info_per_step = 0

        if "no_crashes" in config:
            self.no_crashes = config["no_crashes"]
        else:
            self.no_crashes = False

        # Actions we can take: left, stay, right
        self.action_space = spaces.Discrete(3)

        # verbose level
        verbose_level = config["verbose_level"]
        if verbose_level == 0:
            logging.basicConfig(level=logging.WARNING)
        elif verbose_level == 1:
            logging.basicConfig(level=logging.INFO)
        elif verbose_level == 2:
            logging.basicConfig(level=logging.DEBUG)

        self.current_time = str(datetime.datetime.now())
        self.episode_counter = 0
        self.step_counter = 0

        # DYNAMIC VARIABLES
        logging.info("initialisation" + self.current_time + str(self.episode_counter))
        self.current_object_sizes = None

        agent_config = config["agent"]
        world_config = config["world"]
        drift_config = world_config["drift"]
        objects_config = world_config["objects"]

        # random x position of agent
        if agent_config["initial_x_position"] is None:
            size = agent_config["size"]
            x_width = world_config["x_width"]
            self.x_position_of_agent = rnd.randint(size, x_width - size + 1)
        else:
            self.x_position_of_agent = agent_config["initial_x_position"]

        self.y_position_of_agent = agent_config["size"]

        # objects list includes the absolute x and y position of the objects + its size

        # list of free ranges says where no objects are defined --> used for defining the funnel
        # the lists in this free ranges list consist of two numbers:
        # where a free range starts and where it ends
        # the numbers are included in this range ([x,y] and not (x,y))

        # setup of the list of drift ranges is similar to the free ranges lists but is independent from funnels and objects
        (
            object_range_list,
            list_of_free_ranges,
            list_of_drift_ranges_with_drift_number,
        ) = hlp.create_ranges_of_objects_funnels_and_drifts(
            world_x_width=world_config["x_width"],
            world_y_height=world_config["y_height"],
            height_padding_areas=agent_config["observation_height"],
            level_difficulty=world_config["difficulty"],
            agent_size=agent_config["size"],
            drift_length=drift_config["length"],
            use_variable_drift_intensity=drift_config["variable_intensity"],
            invisible_drift_probability=drift_config["invisible_drift_probability"],
            fake_drift_probability=drift_config["fake_drift_probability"],
        )

        drift_at_whole_level = drift_config["drift_at_whole_level"]
        if drift_at_whole_level == "ranges":
            self.drift_ranges_with_drift_number = list_of_drift_ranges_with_drift_number
        elif drift_at_whole_level == "empty":
            self.drift_ranges_with_drift_number = []
        elif drift_at_whole_level == "left":
            # start, stop, intensity, visibility, fake
            self.drift_ranges_with_drift_number = [
                [1, world_config["y_height"], 1, False, False]
            ]
        elif drift_at_whole_level == "right":
            # start, stop, intensity, visibility, fake
            self.drift_ranges_with_drift_number = [
                [1, world_config["y_height"], -1, False, False]
            ]

        logging.info("object_range_list" + str(object_range_list))
        logging.info("free ranges" + str(list_of_free_ranges))
        logging.info("drift ranges" + str(self.drift_ranges_with_drift_number))

        if (
                objects_config["type"] == "coin"
                and self.reward_function == "pos_neg"
                and world_config["difficulty"] == "hard"
        ):
            number_of_objects = 30
        else:
            number_of_objects = None

        ### OBJECTS
        object_dict_list = hlp.create_list_of_object_dicts(
            object_range_list=object_range_list,
            object_size=agent_config["size"],
            world_x_width=world_config["x_width"],
            level_difficulty=world_config["difficulty"],
            normalized_object_placement=objects_config["normalized_placement"],
            allow_overlapping_objects=objects_config["allow_overlap"],
            number_of_objects=number_of_objects,
        )
        self.object_dict_list = object_dict_list

        ### WALLS
        # the walls are always the same with the same input arguments
        # it doesn't change when resetting the environment!
        # the walls_dict defines for each absolute y value in the world, where the walls are (funnel or not)
        # when the value of the key y is [0, world_x_width] then there is no funnel
        # smaller values define a funnel, e.g. [1, world_x_width - 1] indicate that the wall is one step indented
        walls_dict = hlp.create_dict_of_world_walls(
            list_of_free_ranges=list_of_free_ranges,
            world_y_height=world_config["y_height"],
            world_x_width=world_config["x_width"],
            agent_size=agent_config["size"],
        )
        self.walls_dict = walls_dict
        logging.info("walls_dict" + str(walls_dict))

        self.crashed = False
        self.following_observations_size = min(
            agent_config["observation_height"],
            int(world_config["y_height"] - self.y_position_of_agent + 1),
        )

        self.update_observation()
        # for rendering
        plt.ion()
        self.fig, self.ax = plt.subplots()
        eximg = np.zeros((self.state.shape))
        eximg[0] = -10
        eximg[1] = 3
        self.im = self.ax.imshow(eximg)

        # INITIAL STATE
        self.observation_space = spaces.Box(
            low=-10,
            high=3,
            shape=(self.following_observations_size * (world_config["x_width"] + 2),),
            dtype=np.int64,
        )

        self.information_for_each_step = [[self.state, "Nan", "Nan"]]
        # save all x and y positions of the agent + action
        self.positions_and_action = [
            [int(self.x_position_of_agent), int(self.y_position_of_agent), 1]
        ]

        ### LOGGING
        if verbose_level > 0:
            os.mkdir(self.ROOT_DIR + "/logs/" + self.current_time)

            ### OBJECTS
            self.filepath_for_object_list = (
                    self.ROOT_DIR + "/logs/" + self.current_time + "/object_list.csv"
            )
            # write the objects list of each episode to file
            with open(self.filepath_for_object_list, "a") as file:
                writer = csv.writer(file)
                writer.writerow([self.episode_counter, self.object_dict_list])

            ### WALLS
            self.filepath_for_walls_dict = (
                    self.ROOT_DIR + "/logs/" + self.current_time + "/walls_dict.csv"
            )
            # write the walls definition to file --> same for every episode
            if not os.path.exists(self.filepath_for_walls_dict):
                with open(self.filepath_for_walls_dict, "w") as file:
                    writer = csv.writer(file)
                    writer.writerow([self.walls_dict])

            ### DRIFT
            self.filepath_for_drift_ranges_list = (
                    self.ROOT_DIR + "/logs/" + self.current_time + "/drift_ranges.csv"
            )
            # write the drift ranges of each episode to file
            with open(self.filepath_for_drift_ranges_list, "a") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [self.episode_counter, self.drift_ranges_with_drift_number]
                )

            ### LOGGING EVERY EPISODE
            if verbose_level == 2:
                self.filepath = (
                        self.ROOT_DIR
                        + "/logs/"
                        + self.current_time
                        + "/"
                        + str(self.episode_counter)
                        + ".csv"
                )

                self.filepath_for_vis = (
                        self.ROOT_DIR
                        + "/logs/"
                        + self.current_time
                        + "/"
                        + str(self.episode_counter)
                        + "_vis.csv"
                )

                # write the initial state to file
                with open(self.filepath, "a") as file:
                    writer = csv.writer(file)
                    writer.writerow(["state", "reward", "done"])
                    writer.writerow(self.information_for_each_step[0])

                # write initial state to file
                with open(self.filepath_for_vis, "a") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ["x_position_of_agent", "y_position_of_agent", "action"]
                    )
                    writer.writerow(self.positions_and_action[0])

    def is_done(self) -> bool:
        """
        checks if agent is done by going through the world
        either by crashing in the wall or in an obstacle or by being at the end
        Returns: bool, if agent is done

        """
        return (
                self.crashed
                or self.y_position_of_agent + self.config["agent"]["observation_height"]
                == self.config["world"]["y_height"]
        )

    def apply_action(self, action: int, step_width: int) -> None:
        """
        applies an action 0,1, or 2 (left, stay, right) and updates the x position + the widths to the walls
        Args:
            action (int): integer representing left or right step or staying on same position
            step_width: input_noise added to action movement

        """
        # action 0 - 1 = -1 --> left
        # action 1 - 1 = 0 --> stay
        # action 2 -1 = 1 --> right
        self.crashed = False
        # go down --> go one step further --> y-position changes
        self.y_position_of_agent = int(self.y_position_of_agent) + 1
        self.following_observations_size = min(
            self.config["agent"]["observation_height"],
            int(self.config["world"]["y_height"] - self.y_position_of_agent + 1),
        )

        # action_movement is -1 to go left, 0 to stay and 1 to go right.
        # this allows simply adding the drift force to the action step to compute the
        # next location.
        action_movement = 2 * action - 2 + step_width

        # Pick out the first drift range that contains the current y position, and take its drift direction value
        (_, _, drift, _, is_drift_fake) = next(
            filter(
                lambda drift_range: drift_range[0]
                                    <= self.y_position_of_agent
                                    <= drift_range[1],
                self.drift_ranges_with_drift_number,
            ),
            [0, 0, 0, True, False],
        )

        # Only apply drift of intensity n at every nth step
        if (
                not is_drift_fake
                and drift != 0
                and self.y_position_of_agent % abs(drift) == 0
        ):
            # Keep the direction (sign) but only move one step in the specified direction.
            # The magnitude indicates the intensity of the drift, but we don't have to
            # consider that here as the drift dict already leaves rows without drift if
            # required by the intensity.
            drift_movement = int(math.copysign(1, drift))
        else:
            # This is a non-moving step, drift contributes nothing
            drift_movement = 0

        self.x_position_of_agent += action_movement + drift_movement

        # Clamp x position to the allowed range, to avoid the agents clipping out of bounds when
        # a strong drift occurs and the agent simultaneously takes a step.
        if self.x_position_of_agent < self.config["agent"]["size"]:
            self.x_position_of_agent = self.config["agent"]["size"]
        elif (
                self.x_position_of_agent
                > self.config["world"]["x_width"] + 1 - self.config["agent"]["size"]
        ):
            self.x_position_of_agent = (
                    self.config["world"]["x_width"] + 1 - self.config["agent"]["size"]
            )

    def update_observation(self) -> None:
        """
        generates the new observation/state based on the current environment parameters (agent position, objects,
        drift, etc.)
        """
        self.state = hlp.create_agent_observation(
            following_observation_size=self.following_observations_size,
            drift_ranges=self.drift_ranges_with_drift_number,
            walls_dict=self.walls_dict,
            object_dict_list=self.object_dict_list,
            agent_x_position=self.x_position_of_agent,
            agent_y_position=self.y_position_of_agent,
            world_x_width=self.config["world"]["x_width"],
            agent_size=self.config["agent"]["size"],
            object_type=self.config["world"]["objects"]["type"],
        )

    def calculate_reward(self) -> int:
        """
        calculates reward if the agent has crashed in the wall or in an obstacle
        if agent is in obstacle or wall, reward is -100 & crashed is True
        if agent is near an obstacle or wall, reward is 0
        if agent does a successful step, reward is 10
        Returns (int): reward for the current step

        """
        # state is current observation
        # check for crash if crashes are allowed:
        if not self.no_crashes:
            if self.config["world"]["objects"]["type"] == "obstacle":
                # agent is in obstacle or wall = crash
                if (
                        self.has_agent_collided_with_wall()
                        or len(self.find_intersections(self.object_dict_list)) > 0
                ):
                    self.crashed = True
                    if self.reward_function == "simple":
                        return -100
                    elif self.reward_function == "gaussian":
                        return -1000
                    elif self.reward_function == "pos_neg":
                        raise ValueError(
                            "Reward function {} can not be used with crashes".format(
                                self.reward_function
                            )
                        )

            else:
                # agent is in wall = crash
                if self.has_agent_collided_with_wall():
                    self.crashed = True
                    if self.reward_function == "simple":
                        return -100
                    elif self.reward_function == "gaussian":
                        return -1000
                    elif self.reward_function == "pos_neg":
                        raise ValueError(
                            "Reward function {} can not be used with crashes".format(
                                self.reward_function
                            )
                        )
                    else:
                        raise ValueError(
                            "Reward function {} not implemented".format(
                                self.reward_function
                            )
                        )

        if self.reward_function == "simple":
            relevant_shortened_state = list()
            for row in self.state[0: 2 * self.config["agent"]["size"]]:
                relevant_shortened_state.append(
                    row[
                    max(
                        self.x_position_of_agent - self.config["agent"]["size"], 0
                    ): min(
                        self.x_position_of_agent + self.config["agent"]["size"] + 1,
                        self.state.shape[1],
                    )
                    ]
                )
            relevant_shortened_state = np.array(relevant_shortened_state)

            if self.config["world"]["objects"]["type"] == "obstacle":
                if -1 in relevant_shortened_state:
                    return 0
                return 10
            else:
                collected_coins = self.find_intersections(self.object_dict_list)
                if len(collected_coins) > 0:
                    # Prevent coins from being collected multiple times
                    for coin in collected_coins:
                        self.object_dict_list.remove(coin)
                    return len(collected_coins) * 10

                return 0
        elif self.reward_function == "gaussian":
            # remove agent from state
            blurred_state = copy.deepcopy(self.state)
            blurred_state[blurred_state == 1] = 0

            range_of_agent = range(
                -(math.floor(self.config["agent"]["size"] / 2)),
                (math.floor(self.config["agent"]["size"] / 2)) + 1,
            )
            if self.config["world"]["objects"]["type"] == "coin":
                collected_coins = self.find_intersections(self.object_dict_list)
                if len(collected_coins) > 0:
                    # Prevent coins from being collected multiple times
                    for coin in collected_coins:
                        self.object_dict_list.remove(coin)

                        # find positions where agent is on coin --> only last row of agent is possible
                        row = 2 * self.config["agent"]["size"] - 1

                        x_positions_of_agent = []
                        for index in range_of_agent:
                            x_positions_of_agent.append(
                                self.x_position_of_agent + index
                            )

                        range_of_coin = range(
                            -(math.floor(coin["size"] / 2)),
                            (math.floor(coin["size"] / 2) + 1),
                        )
                        x_positions_of_coin = []
                        for index in range_of_coin:
                            x_positions_of_coin.append(coin["x"] + index)
                        x_positions_where_agent_is_on_coin = list(
                            set(x_positions_of_agent).intersection(x_positions_of_coin)
                        )

                        # replace values of intersection of agent and coin with 2
                        blurred_state[row - 1, x_positions_where_agent_is_on_coin] = 2

                # replace -1 with 255
                blurred_state[blurred_state == -1] = 255
                # replace 0 with 127
                blurred_state[blurred_state == 0] = 127
                # replace 2 with 0
                blurred_state[blurred_state == 2] = 0

            # when obstacles are present, -1 is automatically replaced with 255 and 0 stays 0 when forming to np.uint8
            # form state to np.uint8
            blurred_state = np.asarray(blurred_state, dtype=np.uint8)

            # apply gaussian filter (7x7)
            blurred_state = cv2.GaussianBlur(blurred_state, (7, 7), 0)

            # get values of each pixel of current agent position
            values_of_agent_position = []
            # rows
            for i in range(2 * self.config["agent"]["size"] - 1):
                # columns
                if len(range_of_agent) == 0:
                    values_of_agent_position.append(
                        blurred_state[i][self.x_position_of_agent]
                    )
                else:
                    for j in range_of_agent:
                        values_of_agent_position.append(
                            blurred_state[i][self.x_position_of_agent - j]
                        )

            # calculate reward
            reward = 0
            for value in values_of_agent_position:
                reward += abs(value - 255)
            # normalize reward
            if self.config["world"]["objects"]["type"] == "coin":
                # the reward when no coin is near is 128*9=1152 for a 2-sized agent --> this should be 0
                # the lowest reward possible when collecting a coin for is 1275 a 2-sized agent --> this should be 500
                # the function for this is: f(x) = 0.246x + 1152
                # we calculate the corresponding normalized reward x for the current reward f(x)
                # for reward of 10 when no coin is near:
                # normalized_reward = (reward - 1149.5) / 0.25
                normalized_reward = ((reward - 1152) / 0.246) / 10
            else:
                # we want the same distance as before 10 for successful step, 0 if near an obstacle (obstacle task)
                # the biggest reward possible when near an obstacle is 2219 for a 2-sized agent --> this should be 0
                # the highest reward possible when not near an obstacle is 255 for a 2-sized agent --> this should be 10
                # the function for this is: f(x) = 7.6x + 2219
                # we calculate the corresponding normalized reward x for the current reward f(x)
                normalized_reward = (reward - 2219) / 7.6
            return int(normalized_reward)
            # TODO: test for image state
        elif self.reward_function == "pos_neg":
            if (
                    self.config["world"]["difficulty"] != "easy"
                    and self.config["world"]["difficulty"] != "hard"
            ):
                raise ValueError(
                    "Reward function {} can only be used with easy and hard difficulty".format(
                        self.reward_function
                    )
                )
            # positive reward because of passing
            reward = 0
            self.pos_neg_reward_info_dict_per_step["pos"] = [0]
            self.pos_neg_reward_info_dict_per_step["neg"] = [0]
            for object in self.object_dict_list:
                if (
                        object["y"] + (2 * self.config["agent"]["size"] - 1)
                        == self.y_position_of_agent
                        and object not in self.already_crashed_objects
                ):
                    if self.config["world"]["difficulty"] == "easy":
                        if self.config["world"]["objects"]["type"] == "coin":
                            reward -= 7
                            if self.pos_neg_reward_info_dict_per_step["neg"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["neg"].append(-7)
                            else:
                                self.pos_neg_reward_info_dict_per_step["neg"] = [-7]
                        else:
                            reward += 7
                            if self.pos_neg_reward_info_dict_per_step["pos"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["pos"].append(7)
                            else:
                                self.pos_neg_reward_info_dict_per_step["pos"] = [7]
                    elif self.config["world"]["difficulty"] == "hard":
                        if self.config["world"]["objects"]["type"] == "coin":
                            reward -= 3
                            if self.pos_neg_reward_info_dict_per_step["neg"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["neg"].append(-3)
                            else:
                                self.pos_neg_reward_info_dict_per_step["neg"] = [-3]
                        else:
                            reward += 1
                            if self.pos_neg_reward_info_dict_per_step["pos"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["pos"].append(1)
                            else:
                                self.pos_neg_reward_info_dict_per_step["pos"] = [1]
            for crash in self.find_intersections(self.object_dict_list):
                if crash not in self.already_crashed_objects:
                    if self.config["world"]["difficulty"] == "easy":
                        if self.config["world"]["objects"]["type"] == "coin":
                            reward += 7
                            if self.pos_neg_reward_info_dict_per_step["pos"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["pos"].append(7)
                            else:
                                self.pos_neg_reward_info_dict_per_step["pos"] = [7]
                        else:
                            reward -= 7
                            if self.pos_neg_reward_info_dict_per_step["neg"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["neg"].append(-7)
                            else:
                                self.pos_neg_reward_info_dict_per_step["neg"] = [-7]
                    elif self.config["world"]["difficulty"] == "hard":
                        if self.config["world"]["objects"]["type"] == "coin":
                            reward += 3
                            if self.pos_neg_reward_info_dict_per_step["pos"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["pos"].append(3)
                            else:
                                self.pos_neg_reward_info_dict_per_step["pos"] = [3]
                        else:
                            reward -= 3
                            if self.pos_neg_reward_info_dict_per_step["neg"] != [0]:
                                self.pos_neg_reward_info_dict_per_step["neg"].append(-3)
                            else:
                                self.pos_neg_reward_info_dict_per_step["neg"] = [-3]

                    self.already_crashed_objects.append(crash)

            return reward
        else:
            raise ValueError(
                "Reward function {} not implemented".format(self.reward_function)
            )

    def has_agent_collided_with_wall(self) -> bool:
        """
        Returns: Whether the agent has collided with a wall.
        """
        size = self.config["agent"]["size"]
        # We have to check each row of the agent because there may be
        # varying levels of wall indentation (like in a funnel)
        for agent_y_index in range(size * 2 - 1):
            # - 1 because the left wall isn't included in the world width
            wall_index = str(self.y_position_of_agent - (size - 1) + agent_y_index)
            if (
                    self.x_position_of_agent - size < self.walls_dict[wall_index][0]
                    or self.x_position_of_agent + size - 1
                    > self.walls_dict[wall_index][1] - 1
            ):
                return True
        return False

    def find_intersections(self, objects: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """
        Returns: The list of obstacles/coins intersecting with the agent

        Args:
            objects: List of objects (obstacles/coins/...) to check for an intersection.
        """

        def collides_with_agent(obj) -> bool:
            # Check for collisions by ensuring that the center of the agent is not in a radius around the
            # object equal to the size of the object, plus the size of the agent (-2 because we do want to allow
            # them to be exactly side by side, and they are minimum width 1 each)
            radius = obj["size"] + self.config["agent"]["size"] - 2
            return (
                    abs(self.y_position_of_agent - obj["y"]) <= radius
                    and abs(self.x_position_of_agent - obj["x"]) <= radius
            )

        return list(filter(collides_with_agent, objects))

    def step(self, action: int, step_width: int = 0):
        """
        performs a whole step of an agent including applying an action, updating the observation and getting a reward
        Args:
            action (int): integer representing left or right step or staying on same position
            step_width: input_noise added to action movement

        Returns:
            the state in form of the observation matrix
            the reward
            if the agent is done going through the world
            an empty info dictionary

        """
        # logging.info("step in env")
        if self.is_done():
            raise EnvironmentError(
                "no more action steps possible at current position in the environment"
            )

        # APPLY ACTION
        self.apply_action(action=action, step_width=step_width)

        # UPDATE OBSERVATION
        self.update_observation()

        # set placeholder for truncated
        truncated = False

        # CALCULATE REWARD
        reward = self.calculate_reward()

        # info of rewards
        info = {"simple": self.simple_reward_info_per_step, "gaussian": self.gaussian_reward_info_per_step,
                "pos_neg": self.pos_neg_reward_info_dict_per_step}

        self.positions_and_action = self.positions_and_action + [
            [
                self.x_position_of_agent,
                self.y_position_of_agent,
                action,
            ]
        ]
        self.information_for_each_step = self.information_for_each_step + [
            [self.state, reward, self.is_done()]
        ]

        # at the end of the episode, write log files
        if self.config["verbose_level"] == 2 and self.is_done():
            # write the current step of the agent to the file
            with open(self.filepath, "a") as file:
                writer = csv.writer(file)
                # first element is already added in the initialization
                for step in self.information_for_each_step[1:]:
                    writer.writerow(step)

            ### VISUALIZATION
            with open(self.filepath_for_vis, "a") as file:
                writer = csv.writer(file)
                # first element is already added in the initialization
                for step in self.positions_and_action[1:]:
                    writer.writerow(step)

        self.step_counter += 1
        # return step information
        return self.state.flatten(), reward, self.is_done(), truncated, info

    def render(self):
        self.im.set_data(self.state)
        if self.render_mode == "human":
            self.fig.canvas.draw_idle()
        elif self.render_mode == "rgb_array":
            # read ascii text from numpy array
            img_state = self.state.copy()
            ascii_text = str(img_state)
            # Create a new Image
            # make sure the dimensions (W and H) are big enough for the ascii art
            W, H = (550, 500)
            im = Image.new("RGBA", (W, H), "white")

            # Draw text to image
            draw = ImageDraw.Draw(im)
            _, _, w, h = draw.textbbox((0, 0), ascii_text)
            # draws the text in the center of the image
            draw.text(((W - w) / 2, (H - h) / 2), ascii_text, fill="black")
            return np.array(im)

    def reset(self, seed=None, options=None):
        """
        resets the environment
        """
        # logging.info("reset " + self.current_time + str(self.episode_counter))
        self.episode_counter += 1
        self.step_counter = 0

        agent_config = self.config["agent"]
        world_config = self.config["world"]
        drift_config = world_config["drift"]
        objects_config = world_config["objects"]

        # random x position of agent
        size = agent_config["size"]
        if agent_config["initial_x_position"] is None:
            self.x_position_of_agent = rnd.randint(
                size, world_config["x_width"] - size + 1
            )
        else:
            self.x_position_of_agent = agent_config["initial_x_position"]

        self.y_position_of_agent = size

        (
            object_range_list,
            list_of_free_ranges,
            drift_ranges,
        ) = hlp.create_ranges_of_objects_funnels_and_drifts(
            world_x_width=world_config["x_width"],
            world_y_height=world_config["y_height"],
            height_padding_areas=agent_config["observation_height"],
            level_difficulty=world_config["difficulty"],
            drift_length=drift_config["length"],
            use_variable_drift_intensity=drift_config["variable_intensity"],
            invisible_drift_probability=drift_config["invisible_drift_probability"],
            fake_drift_probability=drift_config["fake_drift_probability"],
        )
        drift_at_whole_level = drift_config["drift_at_whole_level"]
        if drift_at_whole_level == "ranges":
            self.drift_ranges_with_drift_number = drift_ranges
        elif drift_at_whole_level == "no":
            self.drift_ranges_with_drift_number = []
        elif drift_at_whole_level == "left":
            # start, stop, intensity, visibility, fake
            self.drift_ranges_with_drift_number = [
                [1, world_config["y_height"], 1, False, False]
            ]
        elif drift_at_whole_level == "right":
            # start, stop, intensity, visibility, fake
            self.drift_ranges_with_drift_number = [
                [1, world_config["y_height"], -1, False, False]
            ]

        if (
                objects_config["type"] == "coin"
                and self.reward_function == "pos_neg"
                and world_config["difficulty"] == "hard"
        ):
            number_of_objects = 15
        else:
            number_of_objects = None
        ### OBJECTS
        self.object_dict_list = hlp.create_list_of_object_dicts(
            object_range_list=object_range_list,
            object_size=agent_config["size"],
            world_x_width=world_config["x_width"],
            level_difficulty=world_config["difficulty"],
            normalized_object_placement=objects_config["normalized_placement"],
            allow_overlapping_objects=objects_config["allow_overlap"],
            number_of_objects=number_of_objects,
        )

        ### WALLS --> always the same with the same game settings

        self.crashed = False
        self.following_observations_size = min(
            agent_config["observation_height"],
            int(world_config["y_height"] - self.y_position_of_agent + 1),
        )

        self.update_observation()

        # save all x and y positions of the agent + action
        self.positions_and_action = [
            [int(self.x_position_of_agent), int(self.y_position_of_agent), 1]
        ]
        self.information_for_each_step = [[self.state, "Nan", "Nan"]]

        if self.config["verbose_level"] > 0:
            ### OBJECTS
            with open(self.filepath_for_object_list, "a") as file:
                writer = csv.writer(file)
                writer.writerow([self.episode_counter, self.object_dict_list])

            ### DRIFT
            with open(self.filepath_for_drift_ranges_list, "a") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [self.episode_counter, self.drift_ranges_with_drift_number]
                )

            ### LOGGING EVERY EPISODE
            if self.config["verbose_level"] == 2:
                self.filepath = (
                        self.ROOT_DIR
                        + "/logs/"
                        + self.current_time
                        + "/"
                        + str(self.episode_counter)
                        + ".csv"
                )
                self.filepath_for_vis = (
                        self.ROOT_DIR
                        + "/logs/"
                        + self.current_time
                        + "/"
                        + str(self.episode_counter)
                        + "_vis.csv"
                )

        # set placeholder for info
        self.pos_neg_reward_info_dict_per_step = {}
        self.gaussian_reward_info_per_step = 0
        self.simple_reward_info_per_step = 0
        info = {"simple": self.simple_reward_info_per_step, "gaussian": self.gaussian_reward_info_per_step,
                "pos_neg": self.pos_neg_reward_info_dict_per_step}

        return self.state.flatten(), info
