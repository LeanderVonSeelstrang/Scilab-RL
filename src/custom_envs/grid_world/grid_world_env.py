import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, is_it_possible_that_input_noise_is_applied: bool = False):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.is_it_possible_that_input_noise_is_applied = is_it_possible_that_input_noise_is_applied
        # is it in general possible that the agent can encounter input noise?
        if self.is_it_possible_that_input_noise_is_applied:
            # does this episode have input noise?
            self.input_noise_is_applied_in_this_episode = random.choice([True, False])
            print("input noise is applied in this episode:", self.input_noise_is_applied_in_this_episode)

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
        if self.is_it_possible_that_input_noise_is_applied:
            # does this episode have input noise?
            self.input_noise_is_applied_in_this_episode = random.choice([True, False])
            print("input noise is applied in this episode:", self.input_noise_is_applied_in_this_episode)

        # FIXME: not hardcoded
        self._target_location = np.array([4, 0])
        self._agent_location = np.array([0, 4])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3,4,5,6,7}) to the direction we walk in
        if self.input_noise_is_applied_in_this_episode:
            possible_actions = [0, 1, 2, 3, 4, 5, 6, 7]
            weights = [0.5 / 7 for _ in range(len(possible_actions))]
            weights[action] = 0.5
            action = random.choices(population=possible_actions, weights=weights, k=1)[0]
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
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

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # now we draw the traps
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * np.array([1, 4]),
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * np.array([2, 3]),
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * np.array([3, 2]),
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * np.array([4, 1]),
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()