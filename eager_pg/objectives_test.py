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

"""Tests to ensure calculations from objectives are correct."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from eager_pg import objectives
import tensorflow as tf

FIXED_ADVANTAGE = 0.4

# SINGLE TRAJECTORY
TRAJ_LENGTH = 4
N_TRAJS = 1
OBS_DIM = 5
OBSERVED_STATES = np.random.randn(TRAJ_LENGTH, N_TRAJS,
                                  OBS_DIM).astype(np.float32)

# Corresponds to rewards [1, 1, 1, 0] with discount factor 0.99.
OBSERVED_RETURNS = np.array([[2.9701, 1.99, 1, 0]], dtype=np.float32).T
MASKS = np.ones_like(OBSERVED_RETURNS, dtype=np.float32)

# BATCH TRAJECTORY
N_TRAJS_BATCH = 2
OBSERVED_STATES_BATCH = np.random.randn(TRAJ_LENGTH, N_TRAJS_BATCH,
                                        OBS_DIM).astype(np.float32)

OBSERVED_RETURNS_BATCH = np.array(
    [  # Last dim entry of this trajectory should be masked.
        [2.9701, 1.99, 1, 99999],
        # These returns correspond to a sequence of rewards of [1, 1, 2, 2].
        [5.890798, 4.9402, 3.98, 2]
    ],
    dtype=np.float32).T
MASKS_BATCH = np.ones_like(OBSERVED_RETURNS_BATCH, dtype=np.float32)
MASKS_BATCH[-1, 0] = 0  # Mask the last timestep of the first trajectory.


class RandomPolicy(object):
  """A policy that picks actions uniformly randomly."""

  def __call__(self, state_placeholder):
    batch_size = tf.shape(state_placeholder)[1]
    trajectory_length = tf.shape(state_placeholder)[0]
    # two actions with equal probability
    return tf.log(tf.ones([trajectory_length, batch_size], tf.float32) * 0.5)


class ObjectiveTest(tf.test.TestCase):
  """Helper class to reduce boilerplate code."""

  def calculate_loss(self,
                     objective_to_test,
                     obs_states,
                     obs_returns,
                     obs_masks,
                     only_return_tensor=False):
    policy = RandomPolicy()
    loss_fn = objective_to_test()
    # This policy is not like the one that will be used in the main optimization
    # code. This policy accepts tensors of size (traj_length, batch_size, _).
    # The policies in eager_pf.policies will require inputs of size
    # (batch_size, _).
    log_probs_states = policy(obs_states)
    # TODO(zaf): Redo the calculations for code without baseline.
    advantages = obs_returns - FIXED_ADVANTAGE
    loss = loss_fn(
        log_probs=log_probs_states,
        returns=obs_returns,
        masks=obs_masks,
        advantages=advantages)

    return loss.numpy()

  def calculate_single_trajectory_loss(self, objective_to_test):
    return self.calculate_loss(objective_to_test, OBSERVED_STATES,
                               OBSERVED_RETURNS, MASKS)

  def calculate_two_trajectory_loss(self, objective_to_test):
    return self.calculate_loss(objective_to_test, OBSERVED_STATES_BATCH,
                               OBSERVED_RETURNS_BATCH, MASKS_BATCH)


class REINFORCETest(ObjectiveTest):

  def test_calculations(self):
    """Tests the calculations for a single trajectory."""
    expected_loss = 3.02219
    calculated_loss = self.calculate_single_trajectory_loss(
        objectives.REINFORCE)
    tf.logging.info('REINFORCE calucated %f, expected %f', calculated_loss,
                    expected_loss)
    self.assertTrue(np.allclose(calculated_loss, expected_loss))

  def test_calculations_batch(self):
    """Tests the calculations for a batch of trajectories."""
    expected_loss = (3.29949894183396 + 10.543460377202965) / 2
    calculated_loss = self.calculate_two_trajectory_loss(objectives.REINFORCE)
    tf.logging.info('REINFORCE batch calculated %f, expected %f',
                    calculated_loss, expected_loss)
    self.assertTrue(np.allclose(calculated_loss, expected_loss))


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
