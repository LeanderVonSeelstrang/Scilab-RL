---
layout: default
title: Equip RL-algorithm with forward model
parent: Wiki
has_children: false
nav_order: 2
---

You can extend any algorithm with a forward model that learns the environments transition dynamics, i.e., that predicts the `next_observation`, given `observation` and `action`.

# Necessary steps

The following basic steps are necessary to equip an algorithm of your choice with a forward model:

1. Add a new algorithm as described in the corresponding section in this wiki.
2. Extend the new algorithms configuration by a field `fwd`, that contains important parameters for the forward model.
3. Import the desired forward model classes from `src.utils.forward_models.py` and a buffer for the training data from `src.utils.fw_utils.py` (you could also use a replay buffer class provided by stable baselines, but we recommend to use our custom class for convenience).
4. Equip your agent (in some cases named 'policy' or 'actor') with getters to retrieve the shape of an `observation` and an `action`, respectively.
5. Instanciate the forward model and the data buffer.
6. Find the appropriate places to collect training data and to call your forward models `train` method.
7. (Optional) Save your model for later use

Here, we exemplary show you how to equip the algorithm `cleansac` with a forward model.

## Adding a new algorithm

Start with copying the folder `src.custom_algorithms.cleansac`. Name your copy `cleansac_fw`. 

Copy the file `conf.algorithm.cleansac.yaml`. Name your copy `cleansac_fw.yaml`. 

IMPORTANT: Remember to change the name of your algorithm class, the import in the `__init__.py` and the name in the `cleansac_fw.yaml`. If you do not know what this means, please read the section about adding a new algorithm first!

## Extending the configuration

In your newly created configuration file `cleansac_fw.yaml`, add a field `fwd`, that contains all parameters for your forward model. This looks as follows:

```
# Original file

name: 'cleansac'

learning_rate: 0.0003
buffer_size: 1_000_000
learning_starts: 1000
batch_size: 256
tau: 0.005
gamma: 0.99
ent_coef: auto
use_her: True
n_critics: 2

# Whether to set future expected discounted cumulative reward to what the critic
# computes or to just set it to 0 if the episode is done. The SB3 implementation sets it to 0 if done,
# which is equivalent to false, so we set this as default here.
ignore_dones_for_qvalue: False

# Usually, the action scale is determined by the action space of the environment. Sometimes, however, it is
# desireable to not max out the full scale and reduce the action intensity. For this purpose,
# action_scale_factor is multiplied with the action space of the environment.
action_scale_factor: 1.0

# Whether to log each observation and action dimension in each step. This might slow down everything, but it can help
# debugging because it logs each action and observation dimension per step. It only holds for training, not for eval.
log_obs_step: False
log_act_step: False

```
```
# Changed file

name: 'cleansac_fw'

learning_rate: 0.0003
buffer_size: 1_000_000
learning_starts: 1000
batch_size: 256
tau: 0.005
gamma: 0.99
ent_coef: auto
use_her: True
n_critics: 2

# Whether to set future expected discounted cumulative reward to what the critic
# computes or to just set it to 0 if the episode is done. The SB3 implementation sets it to 0 if done,
# which is equivalent to false, so we set this as default here.
ignore_dones_for_qvalue: False

# Usually, the action scale is determined by the action space of the environment. Sometimes, however, it is
# desireable to not max out the full scale and reduce the action intensity. For this purpose,
# action_scale_factor is multiplied with the action space of the environment.
action_scale_factor: 1.0

# Whether to log each observation and action dimension in each step. This might slow down everything, but it can help
# debugging because it logs each action and observation dimension per step. It only holds for training, not for eval.
log_obs_step: False
log_act_step: False

fwd:
  #  Parameters for the forward model
  hidden_size: 128
  n_hidden_layers : 2
  activation_func : 'relu'  # can be 'relu', 'tanh' or 'logistic'
  loss_func : 'l2'  # for deterministic models, use l2; for probabilisticMLE model use nll. You can implement your desired loss in src.utils.forward_models.py
  fw_batch_size : 256
  fw_learning_rate : 0.0001
  retrain_every_n_steps : 2000  # can be set to 1 for algos that already train only in every k'th step
  model_save_path : 'your/path/here'

```

### Forward model parameters

1. hidden_size: Number of neurons in each hidden layer.
2. n_hidden_layers: Numer of hidden layers.
3. activation_func: Activation function.
4. loss_func: Empirical risk function.
5. fw_batch_size: Batch size for training the forward model.
6. fw_learning_rate: Learning rate for training the forward model.
7. retrain_every_n_steps: In most cases, it makes sense to retrain the forward model only after n new data points have been collected. Training after every step slows the process down considerably.
8. model_save_path: If you want to save the model or its state_dict.

## Import the desired forward model class and the training data buffer

So far, we provide two basic types of fully connected, dense MLP.

The class `DeterministicForwardModel` is a point estimator for the next observation.

The class `ProbabilisticForwardMLENetwork` learns a normal distribution over the predicted next observation.

Import the class you need and the training data buffer as follows:

```
"""
Imports for the fw models
"""
from src.utils.forward_models import DeterministicForwardNetwork, ProbabilisticForwardMLENetwork
from src.utils.fw_utils import Training_Data
```

## Equip your agent ("policy", "actor") with getters for action and observation shape

For proper initialization of your forward model, you need to provide the correct shapes of `action` and `observation`. Since the agent needs to handle observations and actions anyways, it makes sense, to receive their dimensionality from it.

You can do this by extend your agent (in many algorithms named "policy" or "actor") in the following way:

```
class Actor(nn.Module):
    def __init__(self, env, action_scale_factor=1.0):
        
        (...)

    def get_observation_shape(self, env):
        if isinstance(env.observation_space, spaces.Dict):
            obs_shape = env.observation_space.spaces['observation'].shape[0]
        else:
            obs_shape = np.array(env.observation_space.shape).prod()

        return obs_shape

    def get_action_shape(self, env):

        if isinstance(env.action_space, spaces.Discrete):
            action_shape = env.action_space.n.size
        else:
            action_shape = np.prod(env.action_space.shape)

        return action_shape
```

## Instanciate forward model and data buffer

First, it is crucial that you add `fwd = {}` as an argument in your algorithms initializer:

```
class CLEANSAC_FW:

    (...)

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

            fwd = {}
    ):
```

Now you can instanciate your forward model and the training data buffer. 

Typically, you want the forward model and training data to be attributes of your algorithm instance. In this case, you can simply add the following attributes at the end of your algorithms initialization:

```
class CLEANSAC_FW:

    (...)

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

            fwd = {}
    ):

        (...)

        """
        Forward model and data buffer initialization
        """
        self.fwd = fwd
        self.obs_shape = self.actor.get_observation_shape(self.env)
        self.action_shape = self.actor.get_action_shape(self.env)
        self.forward_model = DeterministicForwardNetwork(self.fwd, self.obs_shape, self.action_shape)
        self.fw_optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=self.learning_rate)

        self.training_data = Training_Data()
```

## Find the appropriate place to collect training data

Training data consists of triples `(observation, action, next_observation)`, which you want to push into your training data buffer.

Those triples are typically available right after an `env.step(action)` was performed, and BEFORE the `last_observation` is overwritten by a new observation.

In `cleansac`, an appropriate place is directly in the `step_env()` method, right before the `last_observation` is overwritten. You collect training data by your training data buffers corresponding method `collect_training_data`:

```
    def step_env(
            self,
            callback: BaseCallback
    ):

        (...)

        # Collect training data for the forward model
        self.training_data.collect_training_data(self._last_obs, action, new_obs)

        self._last_obs = new_obs

        (...)
```
## Train your model

A good place to call your forward models `train` method is typically in your algorithms `learn` method (right after calling the RL-algorithms own `train` method).

Before training your forward model, it is recommended to batch your data using your data buffers `get_dataloader` method. It is also recommended to log the train loss in order to check, whether the forward model actually learns.

Everytime the forward models `train` method is called, it is (re-)trained on the whole training data set collected so far. It would be very inefficient to do this after every step, because the gradient computation and backpropagation is rather costly.

Instead, we recommend to only (re-)train your forward model every n'th step:

```
    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval=None,
    ):
        (...)

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                self.train()

                """
                Forward model training
                """
                if self.num_timesteps % self.fwd['retrain_every_n_steps'] == 0:  # only train every n steps
                    fw_data_loader = self.training_data.get_dataloader()
                    self.forward_model.train(self.fw_optimizer, fw_data_loader)

                    self.logger.record('fwd/train_loss', self.forward_model.get_average_loss(fw_data_loader))

        (...)

```

## Save your forward model

If you want to save your forward model for later usage, you can do so by filling in `model_save_path` in the `cleansac_fw.yaml` and calling the forward models `save_model` or `save_state_dict` method before end of your RL algorithms training, like this:

```
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1
    ):
        
        (...)
            
        self.forward_model.save_model('your_models_name')
        self.forward_model.save_state_dict('your_state_dicts_name')

        callback.on_training_end()

        return self
```

# Functionalities of pre-implemented forward models
