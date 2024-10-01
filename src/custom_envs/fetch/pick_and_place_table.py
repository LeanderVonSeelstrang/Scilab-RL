import os

from gymnasium.utils.ezpickle import EzPickle


from gymnasium_robotics.envs.fetch import MujocoFetchEnv, MujocoPyFetchEnv

from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv

MODEL_XML_PATH = os.path.join("fetch", "pick_and_place.xml")

class MujocoFetchPickAndPlaceOnTableEnv(MujocoFetchPickAndPlaceEnv, EzPickle):
    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

