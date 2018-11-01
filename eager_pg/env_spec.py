# coding=utf-8
# Copyright 2018 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for environment interface with agent / tensorflow.

The EnvSpec object acts as an interface between tensorflow and
the batch_env.BatchEnv from agents. This is important since BatchedEnv does
not do automatic action clipping like regular gym environments. This causes
action out of bounds errors which need to be dealt with before giving an action
to the BatchEnv. This is a simplified version of
https://github.com/tensorflow/models/blob/master/research/pcl_rl/env_spec.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

import numpy as np
import tensorflow as tf

CPU = 'cpu:0'
GPU = 'gpu:0'


class SpaceEnum(enum.Enum):
  discrete = 0
  box = 1


def get_space_information(space):
  """Returns information about a space.

  Args:
    space: A gym.spaces object.

  Returns:
    Information about the space as a tuple:
      - Shape.
      - Space type (integer representing box or discrete).
      - The limits (min and max values).
  """
  if hasattr(space, 'n'):
    return space.n, SpaceEnum.discrete, None
  elif hasattr(space, 'shape'):
    return np.prod(space.shape), SpaceEnum.box, (space.low, space.high)


class EnvSpec(object):
  """An interface between environment and tensorflow."""

  def __init__(self, env):
    """Given an environment returns an EnvSpec object.

    The EnvSpec provides helper functions to convert actions and observations
    between tensorflow and the environment.

    Args:
      env: A BatchEnv that you want to interface with. Note that all envs within
      a single batch env must have the same observation/action space.

    Raises:
      ValueError: If env is not a BatchEnv instance.
    """
    # Figure out observation space.
    self.obs_space = env._envs[0].observation_space  # pylint: disable=protected-access
    self.batch_size = len(env)
    (self.obs_dims,
     self.obs_type,
     self.obs_info) = get_space_information(self.obs_space)

    # Figure out action space.
    self.act_space = env._envs[0].action_space  # pylint: disable=protected-access
    (self.act_dims,
     self.act_type,
     self.act_info) = get_space_information(self.act_space)

    # Repeat assign for backwards compatabillity.
    self.total_obs_dim = self.obs_dims
    self.total_sampled_act_dim = self.act_dims

    self._env = env

  def convert_obs_gym_to_tf(self, obs, featurize=True):  # pylint: disable=unused-argument
    """Converts a gym observation into a tensorflow suitable format.

    Args:
      obs: A numpy array shaped (batch_size, obs_dim) representing an
        observation from a gym environment.
      featurize: A boolean indicating if the observations should be featurized.
        only used for compatibility in this case.

    Returns:
      A tf.Tensor suitable to pass into a network.
    """
    return tf.constant(obs, dtype=tf.float32)

  def convert_act_tf_to_gym(self, act):
    """Convert the output action of a tensorflow computation to gym.

    Args:
      act: A tf.Tensor shaped (batch_size, action_dim) representing the action.

    Returns:
      A numpy array containing the action suitable to execute in a gym
      environment.
    """
    # Clipping must happen explicitly here because BatchEnv will do an explicit
    # check to see if actions are in the action space.
    if self.act_type == SpaceEnum.box:
      low, high = self.act_info
      act = np.clip(act.numpy(), low, high)
    elif self.act_type == SpaceEnum.discrete:
      act = act.numpy().astype(int)
    else:
      raise TypeError('Environment has unsupported action space.')
    return act

  def convert_reward_gym_to_tf(self, reward):
    """Convert a scalar reward into np.float32."""
    return np.float32(reward)


class GPUEnvSpec(EnvSpec):
  """Env spec that automatically moves tensors between gpu and cpu."""

  def convert_obs_gym_to_tf(self, obs, featurize=True):  # pylint: disable=unused-argument
    """Convert observations from numpy to tensorflow on the gpu."""
    with tf.device(GPU):
      return super(GPUEnvSpec, self).convert_obs_gym_to_tf(obs)

  def convert_act_tf_to_gym(self, act):
    """Convert observations from tensorflow on gpu to numpy on cpu."""
    with tf.device(CPU):
      act = tf.identity(act)
    return super(GPUEnvSpec, self).convert_act_tf_to_gym(act)

if __name__ == '__main__':
  pass
