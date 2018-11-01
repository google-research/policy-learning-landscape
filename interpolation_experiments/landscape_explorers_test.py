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

"""Tests for landscape explorers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from eager_pg import batch_env
from eager_pg import env_spec
import landscape_explorers
import multiprocessing_tools
import tensorflow as tf


ENV_NAME = 'MountainCarContinuous-v0'


class LandscapeExplorersTest(tf.test.TestCase):

  def test_landscape_explorer_parallel(self):
    """Test that landscape explorer without multiprocessing works."""

    def create_env():
      """Creates environments and useful things to rollout trajectories."""
      env = batch_env.BatchEnv([gym.make(ENV_NAME) for _ in range(5)])
      spec = env_spec.EnvSpec(env)
      others = {'max_steps_env': 1500, 'n_trajectories': 5}
      return env, spec, others

    action_shape = 1
    observation_shape = 2

    dummy_params1, dummy_params2 = np.random.normal(
        size=(2, (observation_shape * action_shape) + action_shape))

    def theta_constructor(alpha, beta):  # pylint: disable=unused-argument, g-doc-args
      """Returns a numpy vector to simulate parameters.

      Note that beta here doesn't do anything, it exists just to check if
      coefficient unpacking works.
      """
      return alpha * dummy_params1 + (1 - alpha) * dummy_params2

    policy_arguments = {'std': 5.0}

    val = landscape_explorers.landscape_explorer_parallel((
        (0.9, 0.1),  # coefficients
        theta_constructor,
        policy_arguments,
        create_env))
    self.assertTrue(isinstance(val, (np.float32,)))

  def test_integration(self):
    """Test integration of landscape explorers with multiprocessing_toosl."""

    def create_env():
      """Creates environments and useful things to rollout trajectories."""
      env = batch_env.BatchEnv([gym.make(ENV_NAME) for _ in range(5)])
      spec = env_spec.EnvSpec(env)
      others = {'max_steps_env': 1500, 'n_trajectories': 5}
      return env, spec, others

    action_shape = 1
    observation_shape = 2

    dummy_params1, dummy_params2 = np.random.normal(
        size=(2, (observation_shape * action_shape) + action_shape))

    def theta_constructor(alpha, beta):  # pylint: disable=unused-argument, g-doc-args
      """Returns a numpy vector to simulate parameters.

      Note that beta here doesn't do anything, it exists just to check if
      coefficient unpacking works.
      """
      return alpha * dummy_params1 + (1 - alpha) * dummy_params2

    policy_arguments = {'std': 5.0}

    prepared_arguments = [((1.0 / i, 0), theta_constructor, policy_arguments,
                           create_env) for i in range(1, 10)]

    results = multiprocessing_tools.managed_multiprocessing_loop_to_numpy(
        prepared_arguments,
        save_dir=None,
        function_to_execute=landscape_explorers.landscape_explorer_parallel,
        save_every=50,
    )

    self.assertIsNotNone(results)

    # pylint: disable=line-too-long
    results = multiprocessing_tools.managed_multiprocessing_loop_to_ndjson(
        prepared_arguments,
        save_dir=None,
        function_to_execute=landscape_explorers.paired_landscape_explorer_parallel,
        save_every=50,
    )
    # pylint: enable=line-too-long

    self.assertIsNotNone(results)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
