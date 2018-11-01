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

"""Collects trajectories in the environment.

Given a policy and an environment the `collect_trajectories` function
will return a batch of trajectories collected from the environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def collect_trajectories(env, policy, max_steps=1500, env_spec=None):
  """Collects trajectories from the environment.

  Args:
    env: The environment to collect data in a batched way (batch_env.BatchEnv).
    policy: A policies.* object to get actions and association information from.
    max_steps: The maximum number of steps to execute before ending.
    env_spec: An env_spec.EnvSpec specification for how the observations from
      the environment will be converted to tensorflow and actions from the
      policy the environment.

  Returns:
    rewards: A np.ndarray containing rewards obtained by each trajectory.
      Shape is (trajectory_length, batch_size).
    log_probs: A tf.Tensor containing the log_probabilities of each action
      executed in the environment. Shape is (trajectory_length, batch_size).
    masks: A np.ndarray containing the masks to apply during computations
      of returns and objective functions.
      Shape is (trajectory_length, batch_size).
    entropies: A tf.Tensor with the entropy of the distribution at each action
      of each trajectory. Shape is (trajectory_length, batch_size).
  """
  state_t = env.reset()
  if env_spec is not None:
    state_t = env_spec.convert_obs_gym_to_tf(state_t)

  done = False  # At some point this becomes a list of bools.
  # Initialize lists to collect data from environment.
  log_probs = []
  rewards = []
  dones = [np.zeros(state_t.shape[0])]
  entropies = []

  # Main loop to collect data from the environment.
  while not np.all(done):
    if len(rewards) > max_steps:
      break  # Trajectory too long.
    action, log_prob, entropy = policy(state_t)
    if env_spec is not None:
      action = env_spec.convert_act_tf_to_gym(action)
    state_tp1, reward, done, _ = env.step(action)
    log_probs.append(log_prob)
    if env_spec is not None:
      reward = env_spec.convert_reward_gym_to_tf(reward)
    rewards.append(reward)
    dones.append(done)
    entropies.append(entropy)

    if env_spec is not None:
      state_t = env_spec.convert_obs_gym_to_tf(state_tp1)

  # Create masks to be applied during computations.
  # Being "Done" at time step t means that masks need
  # to be applied from time step t+1.
  masks = 1 - np.stack(dones).astype(np.float32)[:-1]
  rewards = np.stack(rewards)
  log_probs = tf.stack(log_probs)
  entropies = tf.stack(entropies)
  return rewards, log_probs, masks, entropies


def pad_and_stack_tensors(iterable, max_trajectory_length=None):
  """Pad and stack a list of tensors/numpy arrays to the longest trajectory.

  For example, if we provide tensors of the following shapes:
  [(10, 5), (8, 5)], we will pad and concatenate them to return a tensor with
  shape (10, 10). Padding will happen at the bottom of the tensors: so in this
  case two rows (of zeros) will be added to the tensor of shape (8,5) to make a
  tensor of shape (10, 5). They will then be concatenated in the order that they
  appear in the list to make the final output tensor of shape (10, 10).

  Args:
    iterable: A list of np.ndarray or tf.Tensor objects of shape
    (trajectory_length, batch_size) to be padded and stacked.
    max_trajectory_length: The maximum length of the trajectory to pad.

  Returns:
    A tf.Tensor or np.ndarray that has been padded and stacked to the length of
    the longest object in the input list.
  """
  if len(iterable) == 1:
    # Save some time by not doing any computation if we have only one iterable.
    return iterable[0]

  padded_repeats = []  # Stores the results to be stacked.
  cast_to_numpy = False  # Decide if we should cast to numpy after.

  if max_trajectory_length is None:
    max_trajectory_length = np.max([x.shape[0] for x in iterable])

  for repeat in iterable:
    if not tf.contrib.framework.is_tensor(repeat):
      repeat = tf.constant(repeat)
      cast_to_numpy = True

    padding_amount = [[0, max_trajectory_length - repeat.shape[0]], [0, 0]]
    padded_repeats.append(tf.pad(repeat, padding_amount, mode='constant'))

  stacked_tensors = tf.concat(padded_repeats, 1)
  if cast_to_numpy:
    return stacked_tensors.numpy()
  else:
    return stacked_tensors


def repeat_collect_trajectories(env,
                                policy,
                                n_trajectories,
                                max_steps=1500,
                                env_spec=None):
  """A more memory efficient trajectory collector.

  The main motivation to using this is reducing memory usage sequentially
  sampling from the env (which is batched) to make up the required number of
  trajectories.

  Args:
    env: A batch_env.BatchEnv object that can be sampled to return data shaped
      (trajectory_length, batch_size).
    policy: The tf.keras model policy to use during rollouts.
    n_trajectories: The number of trajectories you want to collect from the env
      using policy.
    max_steps: The maximum number of steps to execute before ending.
    env_spec: An env_spec.EnvSpec specification for how the observations from
      the environment will be converted to tensorflow and actions from the
      policy the environment.

  Raises:
    ValueError: If are asking for more trajectories than obtained by a perfect
      division: i.e. n_trajectories % batch_size is not zero.
    ValueError: If you are asking for fewer trajectories than obtained by a
      perfect division: i.e. n_trajectories // batch_size < 1.

  Returns:
    rewards: A np.ndarray containing rewards obtained by each trajectory.
      Shape is (trajectory_length, n_trajectories).
    log_probs: A tf.Tensor containing the log_probabilities of each action
      executed in the environment. Shape is (trajectory_length, n_trajectories).
    masks: A np.ndarray containing the masks to apply during computations
      of returns and objective functions.
      Shape is (trajectory_length, n_trajectories).
    entropies: A tf.Tensor with the entropy of the distribution at each action
      of each trajectory. Shape is (trajectory_length, n_trajectories).
  """
  batch_size = len(env)
  if n_trajectories % batch_size != 0:
    print(n_trajectories, batch_size)
    raise ValueError('The number of trajectories asked for is not divisble by '
                     'the number of batches returned by the environment.')
  n_repeats = n_trajectories // batch_size
  if n_repeats < 1:
    raise ValueError('Asking for fewer trajectories than batches available. '
                     'Please make sure that n_trajectories >= len(env)')
  traj_lens = []  # Store the length of the trajectories to compute padding.
  log_probs = []
  rewards = []
  masks = []
  entropies = []
  for _ in range(n_repeats):
    reward, log_prob, mask, entropy = collect_trajectories(env, policy,
                                                           max_steps, env_spec)
    traj_lens.append(len(reward))
    log_probs.append(log_prob)
    rewards.append(reward)
    masks.append(mask)
    entropies.append(entropy)

  max_traj_len = np.max(traj_lens)  # Decide maximum length of all trajectories.

  to_return = []

  for iterable in [rewards, log_probs, masks, entropies]:
    to_return.append(pad_and_stack_tensors(iterable,
                                           max_trajectory_length=max_traj_len))

  return to_return

if __name__ == '__main__':
  pass
