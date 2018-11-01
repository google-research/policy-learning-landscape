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

"""Tests for eager_pg.trajectory_batch_stats.

Note that the explicit .numpy() casting also implicitly checks that the methods
all return tensors and not numpy arrays.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from eager_pg import trajectory_batch_stats

import tensorflow as tf

tbs = trajectory_batch_stats

TEST_MASK = [[1, 1, 1, 1],
             [1, 1, 1, 0],
             [1, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 1, 0, 0]]  # pyformat: disable

# Generally masks will be floats so we can easily multiply tensors.
NP_TEST_MASK = np.array(TEST_MASK, dtype=np.float32)


class TrajectoryBatchStatsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests to ensure that statistics on batches of trajectory are correct."""

  @property
  def expected_lengths(self):
    return tf.constant([3, 5, 2, 1], dtype=tf.float32)

  def test_get_trajectory_lengths(self):
    """Checks if the length of each trajectory in the batch is correct."""

    # pylint: disable=invalid-name
    TF_TEST_MASK = tf.constant(NP_TEST_MASK)
    TF_TEST_MASK_TF_F64 = tf.cast(TF_TEST_MASK, tf.float64)
    NP_TEST_MASK_NP_F64 = NP_TEST_MASK.astype(np.float64)
    ALL_MASKS = [
        TF_TEST_MASK, NP_TEST_MASK, TF_TEST_MASK_TF_F64, NP_TEST_MASK_NP_F64
    ]
    # pylint: enable=invalid-name

    for mask in ALL_MASKS:
      computed_lengths = tbs.get_trajectory_lengths(mask)
      self.assertTrue(np.allclose(computed_lengths, self.expected_lengths))

  def run_without_lengths(self, stats_function, args):
    """Helper function to run stats."""
    return stats_function(*args)

  def run_with_lengths(self, stats_function, args):
    """Helper function to run stats with precomputed lengths."""
    return stats_function(*args, trajectory_lengths=self.expected_lengths)

  @parameterized.named_parameters(
      dict(
          testcase_name='rewards',
          raw_batch=np.array([[1, 2, 3, 4]] * 5).astype(np.float32),
          statistic_function=tbs.reward_summaries,
          expected_results_with_traj={
              'mean_step_reward': (3. / 3 + 10. / 5 + 6. / 2 + 4. / 1) / 4.0,
              'mean_trajectory_reward': (3. + 10. + 6. + 4.) / 4.0,
              'stderr_trajectory_reward': np.sqrt(np.sum(
                  (np.array([3., 10., 6., 4.]) -
                   (3. + 10. + 6. + 4.) / 4.0)**2 / 3) / 4)
          },
          expected_results_no_traj={
              'mean_trajectory_reward': (5 + 10 + 15 + 20) / 4.0,
              'mean_step_reward': (1 + 2 + 3 + 4) / 4.0
          }),
      dict(
          testcase_name='entropies',
          raw_batch=np.array([[1, 2, 3, 4]] * 5).astype(np.float32),
          statistic_function=tbs.entropy_summaries,
          expected_results_with_traj={
              'mean_step_entropy': (3. / 3 + 10. / 5 + 6. / 2 + 4. / 1) / 4.0,
              'mean_trajectory_entropy': (3. + 10. + 6. + 4.) / 4.0
          }),
  )
  def test_calculations(self,
                        raw_batch,
                        statistic_function,
                        expected_results_with_traj,
                        expected_results_no_traj=None):  # pylint: disable=g-doc-args
    """Test calculations of statistc_name on raw_batch using statistic_function.
    """
    stats = []
    stats.append(
        self.run_with_lengths(statistic_function, (raw_batch, NP_TEST_MASK)))
    stats.append(
        self.run_without_lengths(statistic_function, (raw_batch, NP_TEST_MASK)))

    for stat in stats:
      for expected_key in expected_results_with_traj.keys():
        self.assertAllClose(stat[expected_key].numpy(),
                            expected_results_with_traj[expected_key])

    if expected_results_no_traj is not None:
      stat = self.run_without_lengths(statistic_function, (raw_batch,))
      for expected_key in expected_results_no_traj.keys():
        self.assertAllClose(stat[expected_key].numpy(),
                            expected_results_no_traj[expected_key])

  def test_reward_calculations_errors(self):
    """Ensures that the reward calculations return the correct errors."""
    rewards_as_list = [[1, 2, 3, 4]] * 5
    self.assertRaises(TypeError, tbs.reward_summaries, rewards_as_list, None)

    rewards_as_numpy = np.array(rewards_as_list)
    rewards_as_numpy_wrong_shape = np.expand_dims(rewards_as_numpy, 1)
    self.assertRaises(ValueError, tbs.reward_summaries,
                      rewards_as_numpy_wrong_shape, None)

  # TODO(zaf): Find a way to @parameterized this?
  def test_returns_calculations(self):
    test_returns = np.array([[0.125, 1.875, 0.25, 1.5], [0.25, 1.75, 0.5, 1.0],
                             [0.5, 1.5, 1.0, 0.0]])
    stats = tbs.return_summaries(test_returns)
    expected_mean_return = (0.125 + 1.875 + 0.25 + 1.5) / 4.0
    self.assertEqual(stats['mean_trajectory_return'].numpy(),
                     expected_mean_return)

    pop_variance = np.sum((test_returns[0] - expected_mean_return)**2 / 3)
    standard_error = np.sqrt(pop_variance) / np.sqrt(4)

    self.assertTrue(
        np.allclose(stats['stderr_trajectory_return'].numpy(), standard_error))

if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
