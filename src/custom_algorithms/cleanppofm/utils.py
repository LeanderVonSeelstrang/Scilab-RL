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
    ##### PREDICT NEXT REWARDS #####
    # predict next rewards from last state and default action
    reward_estimation = 0
    for i in range(number_of_future_steps):
        if position_predicting:
            positions = []
            for obs_element in obs:
                # agent in observation is marked with 1
                first_index_with_one = np.where(obs_element.cpu() == 1)[0][0] + 1
                positions.append(first_index_with_one)
            positions = torch.tensor(positions, device=device).unsqueeze(1)
            forward_model_prediction_normal_distribution = fm_network(positions, default_action.float())
        else:
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
        first_index_with_one = np.where(obs_element.cpu() == 1)[0][0] + 1
        positions.append(first_index_with_one)
    return torch.tensor(positions, device=device).unsqueeze(1)
