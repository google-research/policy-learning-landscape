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

"""Tests for eager_pg.trajectory_collector and eager_pg.batch_env.

These tests demonstrate the capabilities of eager_pg.batch_env.BatchEnv vs those
that are bundled with agents.tools.batch_env.BatchEnv. Since the agents version
does not take into account which environments are already terminated, taking
subsequent actions after termination may allow it to come back to life. This is
particularly problematic for mujoco environments. The extensions made in
eager_pg.batch_env prevent this from happening by explicitly tracking which envs
have died and will not execute any actions in that environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from eager_pg import batch_env
import tensorflow as tf


MAX_STEPS = 10  # The max number of steps you can execute in the environment.
BATCH_SIZE = 5  # The size of the batch of environments.
EXPECTED_ACTION_SPACE = 1  # The size of the action space.
EXPECTED_OBS_SPACE = 3  # The size of the observation space.


def get_batched_environment(env_name):
  """Returns a batched version of the environment."""
  return batch_env.BatchEnv([gym.make(env_name) for _ in range(BATCH_SIZE)])


class TestBatchEnv(tf.test.TestCase):

  def test_exception_on_mixed_spaces(self):
    """Test that we cant used mixed environments."""
    mixed_envs = [gym.make('CartPole-v0'), gym.make('MountainCarContinuous-v0')]
    self.assertRaises(ValueError, batch_env.BatchEnv, mixed_envs)

  def test_batch_returns_correct_obs_shapes(self):
    """Test that the Env returns correctly shaped batched observations."""
    env = get_batched_environment('Pendulum-v0')
    obs = env.reset()
    self.assertEqual(obs.shape, (BATCH_SIZE, EXPECTED_OBS_SPACE))

  def test_batch_accepts_correct_action_shapes(self):
    """Test that the Env throws an error when given the wrong action shapes."""
    env = get_batched_environment('Pendulum-v0')
    env.reset()
    fake_actions = np.zeros((BATCH_SIZE, EXPECTED_ACTION_SPACE))
    self.assertIsNotNone(env.step(fake_actions))

    fake_actions_wrong_shape = np.zeros((BATCH_SIZE, EXPECTED_ACTION_SPACE + 1))
    with self.assertRaises(ValueError):
      env.step(fake_actions_wrong_shape)

  def test_exception_on_out_of_range_actions(self):
    """Test that an exception is thrown when actions are outside the bounds."""
    env = get_batched_environment('Pendulum-v0')
    env.reset()
    actions_out_of_range = 500 * np.ones((BATCH_SIZE, EXPECTED_ACTION_SPACE))
    with self.assertRaises(ValueError):
      env.step(actions_out_of_range)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
