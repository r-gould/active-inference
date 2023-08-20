# from https://github.com/alec-tschantz/rl-inference

import os

import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

import gym

HALF_CHEETAH_RUN = "HalfCheetahRun"
HALF_CHEETAH_FLIP = "HalfCheetahFlip"
ANT_MAZE = "AntMaze"

class GymEnv(object):
    def __init__(self, env_name, max_episode_len, action_repeat=1, seed=None):
        self._env = self._get_env_object(env_name)
        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat
        self.done = False
        if seed is not None:
            self._env.seed(seed)
        self.t = 0

    def reset(self):
        self.t = 0
        state = self._env.reset()
        self.done = False
        return state

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            state, reward_k, done, info = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_len
            if done:
                self.done = True
                break
        return state, reward, done, info

    def sample_action(self):
        return self._env.action_space.sample()

    def render(self, mode="human"):
        self._env.render(mode)

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def unwrapped(self):
        return self._env

    def _get_env_object(self, env_name):
        if env_name == HALF_CHEETAH_RUN:
            return HalfCheetahRunEnv()
        elif env_name == HALF_CHEETAH_FLIP:
            return HalfCheetahFlipEnv()
        elif env_name == ANT_MAZE:
            return SparseAntEnv()
        else:
            return gym.make(env_name)


class HalfCheetahRunEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_x_torso = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/half_cheetah.xml" % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = obs[0] - 0.1 * (action ** 2).sum()
        done = False
        return obs, reward, done, {}

    def _get_obs(self):
        z_position = self.sim.data.qpos.flat[1:2]
        y_rotation = self.sim.data.qpos.flat[2:3]
        other_positions = self.sim.data.qpos.flat[3:]
        velocities = self.sim.data.qvel.flat

        x_torso = np.copy(self.get_body_com("torso")[0:1])
        average_velocity = (x_torso - self.prev_x_torso) / self.dt
        y_rotation_sin, y_rotation_cos = np.sin(y_rotation), np.cos(y_rotation)

        obs = np.concatenate(
            [
                average_velocity,
                z_position,
                y_rotation_sin,
                y_rotation_cos,
                other_positions,
                velocities,
            ]
        )
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55


class HalfCheetahFlipEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_x_torso = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/half_cheetah.xml" % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = obs[12] - 0.1 * (action ** 2).sum()
        done = False
        return obs, reward, done, {}

    def _get_obs(self):
        z_position = self.sim.data.qpos.flat[1:2]
        y_rotation = self.sim.data.qpos.flat[2:3]
        other_positions = self.sim.data.qpos.flat[3:]
        velocities = self.sim.data.qvel.flat

        x_torso = np.copy(self.get_body_com("torso")[0:1])
        average_velocity = (x_torso - self.prev_x_torso) / self.dt
        y_rotation_sin, y_rotation_cos = np.sin(y_rotation), np.cos(y_rotation)

        obs = np.concatenate(
            [
                average_velocity,
                z_position,
                y_rotation_sin,
                y_rotation_cos,
                other_positions,
                velocities,
            ]
        )
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55

from mujoco_py.generated import const


def get_state_block(state):
    x = state[2].item()
    y = state[3].item()

    if -1 < x < 1:
        x_block = "low"
    elif 1 < x < 3:
        x_block = "mid"
    elif 3 < x < 5:
        x_block = "high"
    else:
        raise Exception

    if -1 < y < 1:
        y_block = "left"
    elif 1 < y < 3:
        y_block = "center"
    elif 3 < y < 5:
        y_block = "right"
    else:
        raise Exception

    if x_block == "low" and y_block == "left":
        return 0
    elif x_block == "low" and y_block == "center":
        return 1
    elif x_block == "low" and y_block == "right":
        return 2
    elif x_block == "mid" and y_block == "right":
        return 3
    elif x_block == "high" and y_block == "right":
        return 4
    elif x_block == "high" and y_block == "center":
        return 5
    elif x_block == "high" and y_block == "left":
        return 6


def rate_buffer(buffer):
    visited_blocks = [get_state_block(state) for state in buffer.states]
    n_unique = len(set(visited_blocks))
    return n_unique


class SparseAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Observation Space:
        - x torso COM velocity
        - y torso COM velocity
        - 15 joint positions
        - 14 joint velocities
        - (optionally, commented for now) 84 contact forces
    """

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/ant_maze.xml" % dir_path, 5)
        utils.EzPickle.__init__(self)

    @property
    def contact_forces(self):
        return np.clip(self.sim.data.cfrc_ext, -1, 1)

    def step(self, action):
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        return obs, 0, False, {}

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocities = self.sim.data.qvel.flat.copy()

        x_torso = np.copy(self.get_body_com("torso")[0:1])
        x_velocity = (x_torso - self.prev_x_torso) / self.dt
        y_torso = np.copy(self.get_body_com("torso")[1:2])
        y_velocity = (y_torso - self.prev_y_torso) / self.dt

        # contact_force = self.contact_forces.flat.copy()
        # return np.concatenate((x_velocity, y_velocity, position, velocities, contact_force))

        return np.concatenate((x_velocity, y_velocity, position, velocities))

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.prev_x_torso = np.copy(self.get_body_com("torso")[0:1])
        self.prev_y_torso = np.copy(self.get_body_com("torso")[1:2])
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = const.CAMERA_TRACKING
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent
        self.viewer.cam.lookat[
            0
        ] += 1  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 1
        self.viewer.cam.lookat[2] += 1
        self.viewer.cam.elevation = -85
        self.viewer.cam.azimuth = 235

    @property
    def tasks(self):
        t = dict()
        return t