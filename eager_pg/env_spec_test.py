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

"""Tests for eager_pg.env_spec.

Test to ensure that env_spec detects spaces and converts actions correctly.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from eager_pg import env_spec
import tensorflow as tf


class FakeBatchEnv(object):
  """FakeBatchEnv with necessary interfaces without the type checking.

  FakeBatchEnv mocks BatchEnv which batches a list of N environments together so
  that when you interact, it returns N observations and actions.
  """

  def __init__(self, envs):
    self._envs = envs

  def __len__(self):
    """Return the number of environments in the batch."""
    return len(self._envs)

  def reset(self):
    """Reset and return observations from the envs."""
    return np.stack([env.reset() for env in self._envs])

  def step(self, actions):
    """Execute the actions within the environments."""
    for env, action in zip(self._envs, actions):
      env.step(action)
    return True


def get_batched_environment(env_name, batch_size):
  """Returns a batched version of the environment."""
  return FakeBatchEnv([gym.make(env_name) for _ in range(batch_size)])


class EnvSpecTest(tf.test.TestCase):

  def test_detect_cartpole(self):
    """Checks if the spaces on CartPole are detected properly."""
    env = get_batched_environment('CartPole-v0', batch_size=5)
    spec = env_spec.EnvSpec(env)

    # Check if the Types are detected properly.
    self.assertEqual(spec.obs_type, env_spec.SpaceEnum.box)
    self.assertEqual(spec.act_type, env_spec.SpaceEnum.discrete)

    # Check if the sizes are detected properly.
    self.assertEqual(spec.total_obs_dim, 4)
    self.assertEqual(spec.total_sampled_act_dim, 2)

  def test_detect_mujoco(self):
    """Checks if the spaces in a MuJoCo env are detected properly."""
    env = get_batched_environment('Pendulum-v0', batch_size=5)
    spec = env_spec.EnvSpec(env)

    # Check if the Types are detected properly.
    self.assertEqual(spec.obs_type, env_spec.SpaceEnum.box)
    self.assertEqual(spec.act_type, env_spec.SpaceEnum.box)

    # Check if the sizes are detected properly.
    self.assertEqual(spec.total_obs_dim, 3)
    self.assertEqual(spec.total_sampled_act_dim, 1)

  def test_observation_conversion(self):
    """Checks if observations are converted correctly from gym to tf."""
    env = get_batched_environment('Pendulum-v0', batch_size=5)
    spec = env_spec.EnvSpec(env)

    # Check dtype conversion for observations from gym to tensorflow.
    obs_gym = env.reset()
    obs_tf = spec.convert_obs_gym_to_tf(obs_gym)
    self.assertIsInstance(obs_tf, tf.Tensor)
    self.assertEqual(obs_tf.dtype, tf.float32)

  def test_action_conversion(self):
    """Checks if actions are converted correctly from tf to gym."""
    env = get_batched_environment('Pendulum-v0', batch_size=5)
    spec = env_spec.EnvSpec(env)
    env.reset()

    # Simulate an action from tensorflow.
    act_tf = tf.random_normal((128, 1))
    act_gym = spec.convert_act_tf_to_gym(act_tf)
    self.assertEqual(act_gym.dtype, np.float32)
    self.assertIsNotNone(env.step(act_gym))  # Action can be executed.

  def test_action_clipping_works(self):
    """Checks if actions are clipped correctly."""
    env = get_batched_environment('Pendulum-v0', batch_size=5)
    spec = env_spec.EnvSpec(env)
    env.reset()

    # This action should not be clipped.
    act_tf_should_not_clip = tf.zeros((128, 1)) + 0.2
    act_processed = spec.convert_act_tf_to_gym(act_tf_should_not_clip)
    self.assertTrue(np.allclose(act_processed, 0.2), '{}'.format(act_processed))
    self.assertIsNotNone(env.step(act_processed))

    # This action should be clipped.
    act_tf_should_clip = tf.ones((128, 1)) * 5000
    act_processed = spec.convert_act_tf_to_gym(act_tf_should_clip)
    self.assertTrue(np.allclose(act_processed, 2))
    self.assertIsNotNone(env.step(act_processed))


# TODO(zaf): If interested in discrete actions, add those tests here.

if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
