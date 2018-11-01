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

"""Tools to calculate statistics for trajectories in a batch of trajectories.

These methods can be used across the project to have consistent and testable
calculations.

All methods work with either tensors or arrays of size
(trajectory_length, batch_size). All methods return a dict of tf.Tensor objects
to allow easy composability. Statistics that can be computed include:
- Trajectory lengths
- Mean episode reward
- Mean reward per step
- Mean episode length
- Standard error in the mean reward
- Mean episode return
- Standard Error in the episode return
- Mean entropy per episode
- Mean entropy per step
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_trajectory_lengths(masks):
  """Returns lengths of each trajectory in the batch.

  Args:
    masks: a tf.Tensor of shape (trajectory_length, batch_size)

  Returns:
    A list of trajectory lengths of shape (batch_size, )
  """
  if not isinstance(masks, tf.Tensor):
    masks = tf.constant(masks)
  if masks.dtype != tf.int32:
    masks = tf.cast(masks, tf.int32)
  return tf.cast(tf.count_nonzero(masks, 0), tf.float32)


def reward_summaries(rewards, masks=None, trajectory_lengths=None):
  """Calculates summary statistics for the rewards in a batch of trajectories.

  Args:
    rewards: np.ndarray of shape (trajectory_length, batch_size) with rewards.
    masks: np.ndarray of shape (trajectory_length, batch_size) with masks.
      A mask of 0 at index (t, n) means the reward at (t, n) will not count.
    trajectory_lengths: The length of each trajectory in the batch. If not given
      it will compute this using the masks.

  Raises:
    TypeError: If masks and rewards are not numpy arrays.
    ValueError: If the shapes do no match/not shape 2

  Returns:
    A dictionary of `tf.Tensor`'s containing the following attributes:
      - `mean_trajectory_reward`: mean reward for the trajectories in the batch.
      - `mean_step_reward`: mean reward per step of the trajectory.
      - `stderr_trajectory_reward`: standard error in the mean reward
         for trajectories in the batch.
  """

  if not isinstance(rewards, np.ndarray):
    raise TypeError('`rewards` needs to be a numpy array '
                    'but got {}.'.format(type(rewards)))
  if len(rewards.shape) != 2:
    raise ValueError('`rewards` needs to be of shape 2.')

  masked_rewards, trajectory_lengths = _get_masked_trajectory(
      rewards, masks, trajectory_lengths)

  batch_size = rewards.shape[1]
  sqrt_size = np.sqrt(batch_size)  # For denominator of the standard error.

  reward_per_trajectory = np.sum(masked_rewards, axis=0)

  mean_trajectory_reward = tf.constant(reward_per_trajectory.mean())
  std_traj_reward = tf.constant(reward_per_trajectory.std(ddof=1) / sqrt_size)

  mean_step_reward_per_traj = _calculate_per_step_statistic(
      tf.constant(reward_per_trajectory), trajectory_lengths)

  return dict(
      mean_trajectory_reward=mean_trajectory_reward,
      stderr_trajectory_reward=std_traj_reward,
      mean_step_reward=mean_step_reward_per_traj)


def return_summaries(returns):
  """Calculates summary statistics for the returns in a batch of trajectories.

  Note: Returns are already masked so no need to explicitly mask them.

  Args:
    returns: np.ndarray of shape (trajectory_length, batch_size) with returns
      per step.

  Raises:
    TypeError: If returns is not a numpy array.
    ValueError: If returns is not of shape (trajectory_length, batch_size)

  Returns:
    A dictionary of `tf.Tensor`'s containing the following attributes:
      - `mean_trajectory_return`: mean return for trajectories in the batch.
      - `stderr_trajectory_return`: standard error in the mean reward
         for trajectories in the batch.
  """
  if not isinstance(returns, np.ndarray):
    raise TypeError('`returns` needs to be a numpy array.')
  if len(returns.shape) != 2:
    raise ValueError('`returns` is not of shape 2.')

  batch_size = returns.shape[1]
  sqrt_size = np.sqrt(batch_size)

  mean_returns = tf.constant(returns[0].mean())
  std_returns = tf.constant(returns[0].std(ddof=1) / sqrt_size)

  return dict(
      mean_trajectory_return=mean_returns, stderr_trajectory_return=std_returns)


def entropy_summaries(entropies, masks=None, trajectory_lengths=None):
  """Calculates summary statistics for the entropy in a batch of trajectories.

  Args:
    entropies: tf.Tensor of shape (trajectory_length, batch_size) with entropy
      per time step.
    masks: np.ndarray of shape (trajectory_length, batch_size) with masks.

    trajectory_lengths: The length of each trajectory in the batch. If not given
      it will compute this using the masks.

  Returns:
    A dictionary of `tf.Tensor`'s containing the following attributes:
      - `mean_trajectory_entropy`: mean return for trajectories in the batch.
      - `mean_step_entropy`: mean entropy per step of the trajectory.
  """

  masked_entropy, trajectory_lengths = _get_masked_trajectory(
      entropies, masks, trajectory_lengths)

  entropy_per_trajectory = tf.reduce_sum(masked_entropy, 0)
  mean_entropy_per_trajectory = tf.reduce_mean(entropy_per_trajectory)

  mean_entropy_per_step = _calculate_per_step_statistic(entropy_per_trajectory,
                                                        trajectory_lengths)

  return dict(
      mean_trajectory_entropy=mean_entropy_per_trajectory,
      mean_step_entropy=mean_entropy_per_step)


def _calculate_per_step_statistic(statistic_per_trajectory, trajectory_lengths):
  """Calculates per step statistics.

  Args:
    statistic_per_trajectory: A tf.Tensor containing the statistic per
      trajectory. Shape (batch_size, ).
    trajectory_lengths: Non-zero tf.Tensor containing the lengths of the
      trajectory. Shape (batch_size, ).

  Raises:
    InvalidArgumentError: If trajectory_lengths is zero.

  Returns:
    A tensor containing the mean per-step statistic.
  """
  tf.assert_none_equal(trajectory_lengths, tf.constant([0.0], dtype=tf.float32))
  statistic_per_step = statistic_per_trajectory / trajectory_lengths
  mean_statistic_per_step = tf.reduce_mean(statistic_per_step)
  return mean_statistic_per_step


def _get_masked_trajectory(unmasked_trajectory_batch, masks,
                           trajectory_lengths):
  """Get masked trajectory stats.

  Handles cases where no masks and trajectory_lengths are given.

  Args:
    unmasked_trajectory_batch: A tf.Tensor containing the unmasked trajectory
    batch of shape (trajectory_length, batch_size)
    masks: Masks for each component of shape (trajectory_length, batch_size).
      If None will be handled automatically.
    trajectory_lengths: Non-zero tf.Tensor containing the lengths of the
      trajectory of shape (batch_size, ).
      If None, will be automatically computed.

  Raises:
    InvalidArgumentError: If trajectory_lengths is zero.

  Returns:
    A tensor containing the mean per-step statistic.
  """
  if masks is None:
    trajectory_lengths = tf.constant(
        [unmasked_trajectory_batch.shape[0]] *
        unmasked_trajectory_batch.shape[1],
        dtype=tf.float32)
    masked_trajectory_batch = unmasked_trajectory_batch

  else:
    masked_trajectory_batch = tf.multiply(unmasked_trajectory_batch,
                                          tf.constant(masks))

    if trajectory_lengths is None:
      trajectory_lengths = get_trajectory_lengths(masks)

  return masked_trajectory_batch, trajectory_lengths


if __name__ == '__main__':
  pass
