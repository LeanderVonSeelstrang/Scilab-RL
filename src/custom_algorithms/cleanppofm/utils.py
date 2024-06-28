import numpy as np
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_obs(obs: dict) -> torch.Tensor:
    """
    Flatten a dict observation of the Gridworld envs.
    Args:
        obs: observation of type dict

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
    else:
        raise NotImplementedError(
            "Flatten observation not implemented for this environment with this observation type.")


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
                                           number_of_future_steps: int = 10,
                                           maximum_number_of_objects: int = 5) -> float:
    """
    Get the reward estimation of the forward model for number_of_future_steps steps.
    Args:
        fm_network: forward model network
        obs: last actual observation
        position_predicting: boolean if the forward model is predicting the position or actual observation
        default_action: action that is executed when not controlling the environment
        number_of_future_steps: number of future steps to predict
        maximum_number_of_objects: the number of objects that are considered in the forward model prediction

    Returns:
        reward estimation of the forward model for number_of_future_steps steps

    """
    # the first obs is a matrix, the following obs are just the positions
    if position_predicting:
        obs = get_position_and_object_positions_of_observation(obs,
                                                               maximum_number_of_objects=maximum_number_of_objects)

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


def get_reward_with_future_reward_estimation_corrective(rewards: torch.Tensor, future_reward_estimation: float,
                                                        prediction_error: float) -> torch.Tensor:
    """
    Get the reward with future reward estimation corrective.
    Args:
        rewards: rewards of last step
        future_reward_estimation: estimation of future rewards
        prediction_error: error between predicted and last actual observation

    Returns:
        reward with future reward estimation corrective

    """
    rewards_with_future_reward_estimation = rewards + future_reward_estimation
    # different reward estimation for positive and negative rewards
    if rewards_with_future_reward_estimation < 0:
        # negative rewards
        reward_with_future_reward_estimation_corrective = rewards_with_future_reward_estimation + (
                abs(rewards_with_future_reward_estimation) * (1 - prediction_error))
    else:
        # positive rewards
        if not prediction_error == 0:
            reward_with_future_reward_estimation_corrective = rewards_with_future_reward_estimation / prediction_error
        # prediction error is zero -> we cannot divide by zero
        # -> give a boost of * 10 = perfect prediction (* 10 is similar range than other rewards)
        else:
            reward_with_future_reward_estimation_corrective = rewards_with_future_reward_estimation * 10
    return reward_with_future_reward_estimation_corrective


def get_position_and_object_positions_of_observation(obs: torch.Tensor,
                                                     maximum_number_of_objects: int = 5) -> torch.Tensor:
    """
    Get the position of the agent and up to maximum_number_of_objects objects in the observation.
    Args:
        obs: observation
        maximum_number_of_objects: the number of objects that are considered in the observation
        # FIXME: note, these are the first objects you get, when going down in the obs and going from left to right
        # FIXME: these are not necessary the nearest objects to the agent

    Returns:
        position of the agent and maximum_number_of_objects objects in the observation
    """
    agent_and_object_positions = []
    for obs_element in obs:
        current_number_of_objects_in_list = 0
        # agent in observation is marked with 1
        first_index_with_one = np.where(obs_element.cpu() == 1)[0][0]

        # object in observation is marked with 2 or 3
        if 2 in obs_element or 3 in obs_element:
            if 2 in obs_element:
                search_value = 2
            else:
                search_value = 3
            x_y_coordinates = []
            indices_with_two_or_three = np.where(obs_element.cpu() == search_value)[0]

            for index in indices_with_two_or_three:
                # break if we have enough objects
                if current_number_of_objects_in_list >= maximum_number_of_objects:
                    break
                # get x and y coordinate of object
                x_coordinate = index % 12
                y_coordinate = math.floor(index / 12)
                x_y_coordinates.append([x_coordinate, y_coordinate])
                current_number_of_objects_in_list += 1

            # add zeros to the list if we have not enough objects
            while current_number_of_objects_in_list < maximum_number_of_objects:
                x_y_coordinates.append([0, 0])
                current_number_of_objects_in_list += 1

            # add agent to object positions
            x_y_coordinates = [[first_index_with_one % 12, math.floor(first_index_with_one / 12)]] + x_y_coordinates
            agent_and_object_positions.append(x_y_coordinates)
        else:  # no object in observation
            # add agent to object positions + zeros for objects
            x_y_coordinates = [[first_index_with_one % 12, math.floor(first_index_with_one / 12)]] + [[0, 0] for i in
                                                                                                      range(
                                                                                                          maximum_number_of_objects)]
            agent_and_object_positions.append(x_y_coordinates)

    agent_and_object_positions_tensor = torch.flatten(torch.tensor(agent_and_object_positions, device=device),
                                                      start_dim=1)

    return agent_and_object_positions_tensor


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


def get_next_observation_moonlander(observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Calculate the next observation in the moonlander environment manually to exclude random observations through input noise.
    Args:
        observations: observations
        actions: actions

    Returns:
        next observation in the moonlander environment without input noise
    """
    next_observation_without_input_noise = observations.clone().detach()

    for index, obs in enumerate(observations):
        # first two elements are agent x and y position
        counter = 2
        # index[0] is agent x position
        # apply action to agent x position
        next_observation_without_input_noise[index][0] += (actions[index] - 1)
        # clip to range of 1 to 10
        next_observation_without_input_noise[index][0] = torch.clamp(next_observation_without_input_noise[index][0], 1,
                                                                     10)

        # check if there is an object that already is on position 0 -> removed
        while counter < 10:
            if not next_observation_without_input_noise[index][counter] == 0 and \
                    next_observation_without_input_noise[index][counter + 1] == 0:
                next_observation_without_input_noise[index][counter] = 0
                next_observation_without_input_noise[index][counter + 1] = 0
            counter += 2

        # apply step to every y position (agent and objects)
        next_observation_without_input_noise[index][1::2] -= 1
        next_observation_without_input_noise[index][1::2] = torch.clamp(
            next_observation_without_input_noise[index][1::2], 0, 10)

    return next_observation_without_input_noise


def calculate_prediction_error(env_name: str, env, next_obs, forward_model_prediction_normal_distribution: torch.normal,
                               position_predicting: bool, maximum_number_of_objects: int = 5) -> float:
    """
    Calculate the prediction error between the next obs and the forward model prediction.
    Args:
        env_name: name of the environment
        env: environment
        next_obs: observation after actually executing the action
        forward_model_prediction_normal_distribution: prediction of next observation by forward model
        position_predicting: if the forward model is predicting the position or actual observation
        maximum_number_of_objects: the number of objects that are considered in the forward model prediction

    Returns:
        prediction error between the next obs and the forward model prediction
    """
    ##### CALCULATE PREDICTION ERROR #####
    # prediction error version one -> standard deviation
    # prediction_error = forward_normal.stddev.mean().item()
    # prediction error version two -> Euclidean distance
    # calculate manually prediction error (Euclidean distance)
    if env_name == "GridWorldEnv":
        # predicted location can only be between 0 and 4
        max_distance_in_gridworld = math.sqrt(((4 - 0) ** 2) + ((4 - 0) ** 2) + ((4 - 0) ** 2) + ((4 - 0) ** 2))
        predicted_location = torch.tensor(
            [min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][0])), 4),
             min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][1])), 4),
             min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][2])), 4),
             min(max(0, round(forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[0][3])), 4)],
            device=device)
        prediction_error = (math.sqrt(torch.sum((predicted_location - next_obs) ** 2))) / max_distance_in_gridworld
    elif env_name == "MoonlanderWorldEnv":
        # we just care for the x position of the moonlander agent, because the y position is always equally to the size of the agent
        if position_predicting:
            positions = get_position_and_object_positions_of_observation(next_obs,
                                                                         maximum_number_of_objects=maximum_number_of_objects)
            # print("actual positions: ", positions)

            # Smallest x position of the agent is the size of the agent
            # Biggest x position of the agent is the width of the moonlander world - the size of the agent
            # Note: you should use vec_env.env_method("get_wrapper_attr", "attribute_name") in Gymnasium v1.0
            first_possible_x_position = env.env_method("get_wrapper_attr", "first_possible_x_position")[0]
            last_possible_x_position = env.env_method("get_wrapper_attr", "last_possible_x_position")[0]
            max_distance_in_moonlander_world = math.sqrt(
                (last_possible_x_position - first_possible_x_position) ** 2)
            predicted_x_position = torch.tensor([min(max(first_possible_x_position,
                                                         forward_model_prediction_normal_distribution.mean.cpu().detach().numpy()[
                                                             0][0]),
                                                     last_possible_x_position)], device=device)
            prediction_error = (math.sqrt(
                torch.sum((predicted_x_position - positions[0]) ** 2))) / max_distance_in_moonlander_world
        # TODO: implement prediction error calculation for moonlander world env with predicting whole observation
        else:
            raise NotImplementedError(
                "Prediction error calculation not implemented for MoonlanderWorldEnv with predicting whole observation!")
    else:
        raise ValueError("Environment not supported")

    return prediction_error
