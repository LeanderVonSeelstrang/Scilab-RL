import os
import time
import importlib
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
# import myosuite
# import gym as old_gym
import gymnasium as gym
# gym.register_envs()
import wandb
import myosuite

from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from custom_envs.register_envs import register_custom_envs
from utils.util import get_git_label, set_global_seeds, get_train_render_schedule, get_eval_render_schedule, \
    avoid_start_learn_before_first_episode_finishes
from utils.mlflow_util import setup_mlflow, get_hyperopt_score, log_params_from_omegaconf_dict
from utils.custom_logger import setup_logger
from utils.custom_callbacks import EarlyStopCallback, EvalCallback, CustomEvalCallback, CustomEvalCallbackMetaAgent
from utils.custom_wrappers import DisplayWrapper, RecordVideo

# make git_label available in hydra
OmegaConf.register_new_resolver("git_label", get_git_label)


def get_env_instance(cfg, logger):
    train_env = gym.make(cfg.env, **cfg.env_kwargs)
    eval_env = gym.make(cfg.env, **cfg.env_kwargs)

    # wrappers for rendering
    train_render_schedule = get_train_render_schedule(cfg.render_freq)
    eval_render_schedule = get_eval_render_schedule(cfg.render_freq, cfg.n_test_rollouts)
    if cfg.render == 'display':
        train_env = DisplayWrapper(train_env,
                                   steps_per_epoch=cfg.eval_after_n_steps,
                                   episode_in_epoch_trigger=train_render_schedule,
                                   metric_keys=cfg.render_metrics_train,
                                   logger=logger)
        eval_env = DisplayWrapper(eval_env,
                                  episode_trigger=eval_render_schedule,
                                  metric_keys=cfg.render_metrics_test,
                                  logger=logger)
    if cfg.render == 'record':
        train_env = RecordVideo(env=train_env,
                                video_folder=logger.get_dir() + "/videos",
                                name_prefix="train",
                                steps_per_epoch=cfg.eval_after_n_steps,
                                episode_in_epoch_trigger=train_render_schedule,
                                metric_keys=cfg.render_metrics_train,
                                video_length=cfg.render_frames_per_clip,
                                logger=logger)
        eval_env = RecordVideo(env=eval_env,
                               video_folder=logger.get_dir() + "/videos",
                               name_prefix="eval",
                               episode_trigger=eval_render_schedule,
                               metric_keys=cfg.render_metrics_test,
                               video_length=cfg.render_frames_per_clip,
                               logger=logger)

    # The following gym wrappers can be added via commandline parameters,
    # e.g. use +flatten_obs=1 to use the FlattenObservation wrapper
    if 'flatten_obs' in cfg and cfg.flatten_obs:
        train_env = gym.wrappers.FlattenObservation(train_env)
        eval_env = gym.wrappers.FlattenObservation(eval_env)

    if 'clip_action' in cfg and cfg.clip_action:
        train_env = gym.wrappers.ClipAction(train_env)
        eval_env = gym.wrappers.ClipAction(eval_env)

    if 'normalize_obs' in cfg and cfg.normalize_obs:
        train_env = gym.wrappers.NormalizeObservation(train_env)
        eval_env = gym.wrappers.NormalizeObservation(eval_env)

    if 'normalize_reward' in cfg and cfg.normalize_reward:
        train_env = gym.wrappers.NormalizeReward(train_env)
        eval_env = gym.wrappers.NormalizeReward(eval_env)

    if 'time_aware_observation' in cfg and cfg.time_aware_observation:
        train_env = gym.wrappers.TimeAwareObservation(train_env)
        eval_env = gym.wrappers.TimeAwareObservation(eval_env)

    # At last, wrap in DummyVecEnv. This has to be the last wrapper, because it breaks the .unwrapped attribute.
    train_env = DummyVecEnv([lambda: train_env])
    eval_env = DummyVecEnv([lambda: eval_env])

    return train_env, eval_env


def get_algo_instance(cfg, logger, env):
    algo_name = cfg['algorithm'].name
    alg_kwargs = OmegaConf.to_container(cfg.algorithm)
    del alg_kwargs['name']  # remove name as we pass all arguments to the model constructor
    try:
        baseline_class = getattr(importlib.import_module('stable_baselines3.' + algo_name), algo_name.upper())
    except ModuleNotFoundError:
        if algo_name == 'recurrentppo':
            # Recurrent PPO is not part of stable-baselines3, but of sb3-contrib
            # BUT PPO with frame-stacking is usually quite competitive if not better, and faster than recurrent PPO!
            # Exception: CarRacing-v0 and LunarLanderNoVel-v2
            baseline_class = getattr(importlib.import_module('sb3_contrib.' + 'ppo_recurrent'), 'RecurrentPPO')
        else:
            baseline_class = getattr(importlib.import_module('custom_algorithms.' + algo_name), algo_name.upper())
    if 'replay_buffer_class' in alg_kwargs and alg_kwargs['replay_buffer_class'] == 'HerReplayBuffer':
        alg_kwargs['replay_buffer_class'] = HerReplayBuffer
        alg_kwargs = avoid_start_learn_before_first_episode_finishes(alg_kwargs, env)
    if cfg.restore_policy is not None:
        baseline = baseline_class.load(cfg.restore_policy, env=env, **alg_kwargs)
    else:
        baseline = baseline_class(env=env, **alg_kwargs)
    baseline.set_logger(logger)
    return baseline


def create_callbacks(cfg, logger, eval_env):
    callback = []

    if cfg.save_model_freq > 0:
        checkpoint_callback = CheckpointCallback(save_freq=cfg.save_model_freq, save_path=logger.get_dir(), verbose=1)
        callback.append(checkpoint_callback)

    if cfg['algorithm'].name == 'cleanppofm':
        eval_callback = CustomEvalCallback(eval_env, n_eval_episodes=cfg.n_test_rollouts,
                                           eval_freq=cfg.eval_after_n_steps,
                                           log_path=logger.get_dir(), best_model_save_path=logger.get_dir(),
                                           render=False,
                                           warn=False)
    elif cfg['env'].startswith('MetaEnv'):
        eval_callback = CustomEvalCallbackMetaAgent(eval_env, n_eval_episodes=cfg.n_test_rollouts,
                                                    eval_freq=cfg.eval_after_n_steps,
                                                    log_path=logger.get_dir(), best_model_save_path=logger.get_dir(),
                                                    render=False,
                                                    warn=False)
    else:
        eval_callback = EvalCallback(eval_env, n_eval_episodes=cfg.n_test_rollouts, eval_freq=cfg.eval_after_n_steps,
                                     log_path=logger.get_dir(), best_model_save_path=logger.get_dir(), render=False,
                                     warn=False)

    callback.append(eval_callback)
    early_stop_callback = EarlyStopCallback(metric=cfg.early_stop_data_column, eval_freq=cfg.eval_after_n_steps,
                                            threshold=cfg.early_stop_threshold, n_episodes=cfg.early_stop_last_n)
    callback.append(early_stop_callback)
    callback = CallbackList(callback)
    return callback


# config_path is relative to the location of the Python script
@hydra.main(config_name="main", config_path="../conf", version_base="1.1.2")
def main(cfg: DictConfig): # -> (float, int):
    run_dir = os.getcwd()
    if cfg.restore_policy is not None:
        run_dir = os.path.split(cfg.restore_policy)[:-1][0]
        run_dir = run_dir + "_restored"
    run_name = cfg['algorithm']['name'] + '_' + cfg['env']

    register_custom_envs()
    setup_mlflow(cfg)

    with mlflow.start_run(run_name=run_name) as mlflow_run:
        mlflow.log_param('log_dir', run_dir)
        logger = setup_logger(run_dir, run_name, cfg)
        run_id = mlflow.active_run().info.run_id
        print(f"Active mlflow run_id: {run_id}")
        log_params_from_omegaconf_dict(cfg)
        OmegaConf.save(config=cfg, f='params.yaml')
        if cfg['seed'] == 0:
            cfg['seed'] = int(time.time_ns() % 2 ** 32)
        set_global_seeds(cfg.seed)

        train_env, eval_env = get_env_instance(cfg, logger)

        baseline = get_algo_instance(cfg, logger, train_env)

        callback = create_callbacks(cfg, logger, eval_env)

        logger.info("Launching training")
        training_finished = False
        total_steps = cfg.eval_after_n_steps * cfg.n_epochs
        try:
            baseline.learn(total_timesteps=total_steps, callback=callback, log_interval=None)
            training_finished = True
            logger.info("Training finished!")
            # Save model when training is finished
            p = logger.get_dir() + "/rl_model_finished"
            logger.info(f"Saving policy to {p}")
            baseline.save(path=p)
        except ValueError as e:
            if e.args[0].startswith("Expected parameter loc"):
                logger.error(f"The experiment failed with error {e}")
                logger.error("If this error happened because of a tensor with NaNs in it, "
                             "that is probably because the chosen hyperparameters made the algorithm unstable.")
            else:
                raise e
        train_env.close()
        eval_env.close()

        # after training
        if training_finished:
            hyperopt_score, n_epochs = get_hyperopt_score(cfg, mlflow_run)
        else:
            hyperopt_score, n_epochs = -1, cfg["n_epochs"]
        logger.info(f"Hyperopt score: {hyperopt_score}, epochs: {n_epochs}.")
        mlflow.log_metric("hyperopt_score", hyperopt_score)
        with open(os.path.join(run_dir, 'train.log'), 'r') as logfile:
            log_text = logfile.read()
            mlflow.log_text(log_text, 'train.log')
        if cfg["wandb"]:
            wandb.log({"hyperopt_score": hyperopt_score})
            wandb.finish()

    return hyperopt_score, n_epochs, run_id


if __name__ == '__main__':
    main()
