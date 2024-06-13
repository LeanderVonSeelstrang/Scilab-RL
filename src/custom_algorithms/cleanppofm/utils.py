import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_obs(obs) -> torch.Tensor:
    """
    Flatten an observation
    Args:
        obs: observation of type ...

    Returns:
        flattened observation in tensor format

    """
    # tensor can not check for string ("agent" in obs)
    if isinstance(obs, dict):
        # this is the flatten process for the Gridworld envs
        agent, target = obs['agent'], obs['target']
        if isinstance(agent, np.ndarray):
            agent = torch.from_numpy(agent).to(device)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).to(device)
        return torch.cat([agent, target], dim=1).to(dtype=torch.float32).detach().clone()
    # RGB image
    # FIXME: missing documentation because of error in moonlander
    else:
        return torch.tensor(obs, device=device, dtype=torch.float32).flatten(start_dim=1).detach().clone()


def layer_init(layer, std: np.float64 = np.sqrt(2), bias_const: float = 0.0) -> torch.nn.Module:
    """
    Initialize the layer with a (semi) orthogonal matrix and a constant bias.
    Args:
        layer: layer to be initialized
        std: standard deviation for the orthogonal matrix
        bias_const: constant for the bias

    Returns:
        initialized layer

    """
    # Fill the layer weight with a (semi) orthogonal matrix.
    # Described in Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    # Saxe, A. et al. (2013).
    torch.nn.init.orthogonal_(layer.weight, std)
    # Fill the layer bias with the value bias_const.
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_reward_estimation_of_forward_model(fm_network, obs: torch.Tensor,
                                           position_predicting: bool,
                                           default_action: torch.Tensor = torch.Tensor([[0]]),
                                           number_of_future_steps: int = 10) -> float:
    """
    Get the reward estimation of the forward model for number_of_future_steps steps.
    Args:
        fm_network: forward model network
        obs: last actual observation
        position_predicting: boolean if the forward model is predicting the position or actual observation
        default_action: action that is executed when not controlling the environment
        number_of_future_steps: number of future steps to predict

    Returns:
        reward estimation of the forward model for number_of_future_steps steps

    """
    # the first obs is a matrix, the following obs are just the positions
    if position_predicting:
        obs = get_position_of_observation(obs)

    ##### PREDICT NEXT REWARDS #####
    # predict next rewards from last state and default action
    reward_estimation = 0
    for i in range(number_of_future_steps):
        forward_model_prediction_normal_distribution = fm_network(obs, default_action.float())
        # add reward estimation to last reward
        reward_estimation += forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][-1]
        ##### REMOVE REWARD FROM NEW PREDICTED OBS #####
        obs = torch.clamp(torch.round(forward_model_prediction_normal_distribution.mean[0][:-1].unsqueeze(dim=0)),
                          min=0,
                          max=4)
    return reward_estimation


def get_position_of_observation(obs: torch.Tensor) -> torch.Tensor:
    """
    Get the position of the agent in the observation.
    Args:
        obs: observation

    Returns:
        position of the agent in the observation

    """
    positions = []
    for obs_element in obs:
        # agent in observation is marked with 1
        first_index_with_one = np.where(obs_element.cpu() == 1)[0][0]
        positions.append(first_index_with_one)
    return torch.tensor(positions, device=device).unsqueeze(1)


def get_next_observation_gridworld(observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Calculate the next observation in the gridworld environment manually to exclude random observations through input noise.
    Args:
        observations: observations
        actions: actions

    Returns:
        next observation in the gridworld environment without input noise

    """
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
            np.array(observations[index][0:2].cpu()) + direction, 0, 4
        )
        agent_location_without_input_noise[index] = torch.tensor(
            np.concatenate((standard_agent_location, observations[index][2:4].cpu())), device=device
        )
    return agent_location_without_input_noise
