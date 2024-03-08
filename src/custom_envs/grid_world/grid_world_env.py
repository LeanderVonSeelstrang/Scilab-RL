import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, is_it_possible_that_input_noise_is_applied: bool = False,
                 scene_of_input_noise: bool = False):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.is_it_possible_that_input_noise_is_applied = is_it_possible_that_input_noise_is_applied
        self.scene_of_input_noise = scene_of_input_noise
        self.metadata["scene_of_input_noise"] = self.scene_of_input_noise
        self.input_noise_is_applied_in_this_episode = False
        # is it in general possible that the agent can encounter input noise?
        if self.is_it_possible_that_input_noise_is_applied:
            # does this episode have input noise?
            self.input_noise_is_applied_in_this_episode = random.choice([True, False])

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 8 actions, corresponding to
        # "right", "right down", "down", "left down", "left", "left up", "up", "right up"
        self.action_space = spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([1, 1]),  # right down (diagonal)
            2: np.array([0, 1]),  # down
            3: np.array([-1, 1]),  # left down (diagonal)
            4: np.array([-1, 0]),  # left
            5: np.array([-1, -1]),  # left up
            6: np.array([0, -1]),  # up
            7: np.array([1, -1])  # right up
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "self.input_noise_is_applied_in_this_episode": self.input_noise_is_applied_in_this_episode
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # is it in general possible that the agent can encounter input noise?
        self.input_noise_is_applied_in_this_episode = False
        if self.is_it_possible_that_input_noise_is_applied:
            # does this episode have input noise?
            self.input_noise_is_applied_in_this_episode = random.choice([True, False])

        # FIXME: not hardcoded
        self._target_location = np.array([4, 0])
        self._agent_location = np.array([0, 4])

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     _render_frame(window=self.window, render_mode=self.render_mode, clock=self.clock,
        #                   window_size=self.window_size, size=self.size, agent_location=self._agent_location,
        #                   target_location=self._target_location)

        return observation, info

    def step(self, action):
        # scene is a 3x3 grid in the upper left corner
        if self.input_noise_is_applied_in_this_episode or (
                self.scene_of_input_noise and self._agent_location[0] < 3 and self._agent_location[1] < 3):
            possible_actions = [0, 1, 2, 3, 4, 5, 6, 7]
            weights = [0.5 / 7 for _ in range(len(possible_actions))]
            weights[action] = 0.5
            action = random.choices(population=possible_actions, weights=weights, k=1)[0]
        self._agent_location = apply_action(action=action, last_agent_location=self._agent_location,
                                            action_to_direction=self._action_to_direction)
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        # reward
        if np.array_equal(self._agent_location, np.array([1, 4])) or np.array_equal(self._agent_location,
                                                                                    np.array([2, 3])) or np.array_equal(
            self._agent_location, np.array([3, 2])) or np.array_equal(self._agent_location,
                                                                      np.array([4, 1])):
            reward = -100
        else:
            reward = 1 if terminated else -1
        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     _render_frame(window=self.window, render_mode=self.render_mode, clock=self.clock,
        #                   window_size=self.window_size, size=self.size, agent_location=self._agent_location,
        #                   target_location=self._target_location)

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return _render_frame(window=self.window, render_mode=self.render_mode, clock=self.clock,
                                 window_size=self.window_size, size=self.size, agent_location=self._agent_location,
                                 target_location=self._target_location)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def _render_frame(agent_location, target_location, predicted_agent_location=None, predicted_target_location=None,
                  window=None, clock=None, window_size=512, render_mode="human",
                  size=5, title="GridWorld", scene_of_input_noise=False, actual_agent_location=None):
    if predicted_agent_location is not None and predicted_target_location is not None:
        range_of_grid = size * 2
        grid_window_size = window_size * 2
    else:
        range_of_grid = size
        grid_window_size = window_size

    if window is None and render_mode == "human":
        pygame.init()
        pygame.display.init()
        window = pygame.display.set_mode(
            (grid_window_size, window_size)
        )
        pygame.display.set_caption(title)
    if clock is None and render_mode == "human":
        clock = pygame.time.Clock()

    canvas = pygame.Surface((grid_window_size, window_size))
    canvas.fill((255, 255, 255))
    pix_square_size = (
            window_size / size
    )  # The size of a single grid square in pixels

    # if scene is applied in this episode, draw the scene
    if scene_of_input_noise:
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[0, 0], color=(136, 136, 136))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[0, 1], color=(136, 136, 136))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[0, 2], color=(136, 136, 136))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[1, 0], color=(136, 136, 136))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[1, 1], color=(136, 136, 136))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[1, 2], color=(136, 136, 136))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[2, 0], color=(136, 136, 136))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[2, 1], color=(136, 136, 136))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[2, 2], color=(136, 136, 136))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[5, 0], color=(110, 110, 110))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[5, 1], color=(110, 110, 110))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[5, 2], color=(110, 110, 110))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[6, 0], color=(110, 110, 110))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[6, 1], color=(110, 110, 110))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[6, 2], color=(110, 110, 110))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[7, 0], color=(110, 110, 110))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[7, 1], color=(110, 110, 110))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[7, 2], color=(110, 110, 110))
    # First we draw the target
    render_rect(canvas=canvas, pix_square_size=pix_square_size, position=target_location, color=(255, 0, 0))
    # now we draw the traps
    render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[1, 4], color=(0, 255, 0))
    render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[2, 3], color=(0, 255, 0))
    render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[3, 2], color=(0, 255, 0))
    render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[4, 1], color=(0, 255, 0))

    # Now we draw the agent
    pygame.draw.circle(
        canvas,
        (0, 0, 255),
        (agent_location + 0.5) * pix_square_size,
        pix_square_size / 3,
    )
    if actual_agent_location is not None:
        if actual_agent_location[0] != agent_location[0] or actual_agent_location[1] != agent_location[1]:
            pygame.draw.circle(
                canvas,
                (255, 165, 0),
                (actual_agent_location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
    if predicted_agent_location is not None and predicted_target_location is not None:
        # First we draw the predicted target
        target_location_1 = np.copy(predicted_target_location)
        target_location_1[0] += 5
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=target_location_1, color=(100, 0, 0))
        # now we draw the traps for the predicted env
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[6, 4], color=(0, 100, 0))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[7, 3], color=(0, 100, 0))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[8, 2], color=(0, 100, 0))
        render_rect(canvas=canvas, pix_square_size=pix_square_size, position=[9, 1], color=(0, 100, 0))

        # Now we draw the predicted agent
        agent_1_location = np.copy(predicted_agent_location)
        agent_1_location[0] += 5
        pygame.draw.circle(
            canvas,
            (0, 0, 200),
            (agent_1_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
    # Finally, add some gridlines
    for x in range(range_of_grid + 1):
        pygame.draw.line(
            canvas,
            0,
            (0, pix_square_size * x),
            (grid_window_size, pix_square_size * x),
            width=3,
        )
        pygame.draw.line(
            canvas,
            0,
            (pix_square_size * x, 0),
            (pix_square_size * x, grid_window_size),
            width=3,
        )

    if render_mode == "human":
        # The following line copies our drawings from `canvas` to the visible window
        window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        # FIXME: not hardcoded
        clock.tick(4)
        return window, clock
    else:  # rgb_array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )


def render_rect(canvas, pix_square_size, position, color):
    pygame.draw.rect(
        canvas,
        color,
        pygame.Rect(
            pix_square_size * np.array(position),
            (pix_square_size, pix_square_size),
        ),
    )


def apply_action(action, last_agent_location, action_to_direction):
    # Map the action (element of {0,1,2,3,4,5,6,7}) to the direction we walk in
    direction = action_to_direction[action]
    # We use `np.clip` to make sure we don't leave the grid
    agent_location = np.clip(
        last_agent_location + direction, 0, 4
    )
    return agent_location
