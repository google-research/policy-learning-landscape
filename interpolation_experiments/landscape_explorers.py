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

"""Some common landscape explorers.


These functions take a tuple of `args` and return values of the objective
function. See individual explorers for their details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from eager_pg import param_utils
from eager_pg import policies
from eager_pg import rl_utils
from eager_pg import trajectory_batch_stats
from eager_pg import trajectory_collector
import tensorflow as tf


tb_stats = trajectory_batch_stats
logging = tf.logging
gfile = tf.gfile
DEF_DISCOUNT = 0.995  # Default discount factor for Mujoco experiments.


def landscape_explorer_parallel(args):
  r"""Evaluates expected reward objective at a point in landscape.

  This is designed to be run in parallel in a ThreadPool. Each evaluation
  creates a fresh copy of the environment and policy and sets the modified
  parameter vectors according to user defined coefficients.

  An example of usage is:
  ```
  def theta_creator(alpha, beta):
    # theta0 and theta2 are defined outside.
    return theta0 * alpha + theta2 * beta

  policy_arguments = dict(std=0.5)

  def create_env():
    # Create a Batched Environment here.
    # Pass it through env_spec.EnvSpec().
    return env, env_spec, dict(max_steps_env=1500)

  arguments = [((x, -x), \
                theta_creator, \
                policy_arguments, \
                create_env) for x in np.linspace(-0.5, 1.5)]

  pool = multiprocessing.pool.ThreadPool()
  pool.map(landscape_explorer_parallel, arguments)
  ```

  Alternatively you can use common_tools.managed_multiprocessing_loop.

  Args:
    args: For the purposes of ThreadPool, arguments are supplied as tuples and
    their index is as follows:
      0 - coeffs (a tuple containing floats for coefficients to create
          new parameter vectors).
      1 - theta_constructor (function that takes in coeffs and returns a numpy
          array with the new parameter vector ex: theta_constructor(*coeffs).
      2 - policy_args (arguments to pass to the policy class constructor).
      3 - create_env (function that returns and environment, env_spec and a
          dict containing "other" information, one of which must be
          `max_steps_env`).

  Returns:
    The mean reward for this parameter configuration.
  """
  coeffs, theta_constructor, policy_args, create_env = args
  theta_dash = theta_constructor(*coeffs)

  env, spec, others = create_env()
  policy = policies.NormalPolicyFixedStd(
      action_space_size=env.action_space.shape[0], **policy_args)

  # Initialize policy parameters by doing a forward pass and exiting.
  trajectory_collector.collect_trajectories(
      env, policy, max_steps=2, env_spec=spec)

  # Set parameter vector.
  policy = param_utils.set_flat_params(policy, theta_dash)

  # Collect trajectories.
  trajectory_batch = trajectory_collector.repeat_collect_trajectories(
      env, policy, n_trajectories=others['n_trajectories'],
      env_spec=spec, max_steps=others['max_steps_env'])
  rewards, _, masks, _ = trajectory_batch

  # Collect trajectory statistics.
  stats = {}
  trajectory_lengths = tb_stats.get_trajectory_lengths(masks)
  stats['mean_trajectory_lengths'] = tf.reduce_mean(trajectory_lengths)
  stats.update(
      tb_stats.reward_summaries(
          rewards, masks, trajectory_lengths=trajectory_lengths))

  mean_trajectory_reward_for_theta_dash = (
      stats['mean_trajectory_reward'].numpy())
  stderr_trajectory_reward_for_theta_dash = (
      stats['stderr_trajectory_reward'].numpy())

  logging.info('coeffs= %s: ER=%.4f, stderr=%.4f, traj_len=%d', coeffs,
               mean_trajectory_reward_for_theta_dash,
               stderr_trajectory_reward_for_theta_dash,
               stats['mean_trajectory_lengths'].numpy())

  return mean_trajectory_reward_for_theta_dash


# pylint: disable=g-doc-args
def paired_landscape_explorer_parallel(args):
  r"""Evaluates expected reward objective at a point in landscape.

  See multiprocessing_tools.landscape_explorer_parallel for more information
  about how this works.

  The main extension in this explorer is that it returns two values that
  represent moving in two directions -coeff and +coeff.

  Args:
    See multiprocessing_tools.landscape_explorer_parallel.
  Returns:
    The mean reward for this parameter configuration.
  """
  # pylint: enable=g-doc-args
  coeffs, theta_constructor, policy_args, create_env = args

  alpha, theta0_shape = coeffs
  random_direction = np.random.normal(size=(int(theta0_shape),))
  unit_random_direction = random_direction / np.linalg.norm(random_direction)
  theta_pos_direction = theta_constructor(alpha, unit_random_direction)
  theta_neg_direction = theta_constructor(-alpha, unit_random_direction)

  mean_trajectory_rewards = []
  results = {}
  for direction, theta_dash in zip(['pos', 'neg'],
                                   [theta_pos_direction, theta_neg_direction]):
    env, spec, others = create_env()
    policy = policies.NormalPolicyFixedStd(
        action_space_size=env.action_space.shape[0], **policy_args)

    # Initialize policy parameters by doing a forward pass and exiting.
    trajectory_collector.collect_trajectories(
        env, policy, max_steps=2, env_spec=spec)

    # Set parameter vector.
    policy = param_utils.set_flat_params(policy, theta_dash)

    # Collect trajectories.
    trajectory_batch = trajectory_collector.repeat_collect_trajectories(
        env, policy, n_trajectories=others['n_trajectories'],
        env_spec=spec, max_steps=others['max_steps_env'])
    rewards, _, masks, _ = trajectory_batch

    # Collect trajectory statistics.
    stats = {}
    trajectory_lengths = tb_stats.get_trajectory_lengths(masks)
    stats['mean_trajectory_lengths'] = tf.reduce_mean(trajectory_lengths)
    returns = rl_utils.compute_discounted_return(rewards, DEF_DISCOUNT, masks)
    stats.update(tb_stats.return_summaries(returns))
    stats.update(
        tb_stats.reward_summaries(
            rewards, masks, trajectory_lengths=trajectory_lengths))

    # Make safe for json dumping
    safe_stats = {}
    for stat_key, stat_value in stats.items():
      safe_stats[stat_key] = np.asscalar(stat_value.numpy())
    results[direction] = safe_stats

    mean_trajectory_reward_for_theta_dash = stats['mean_trajectory_reward'].numpy()  # pylint: disable=line-too-long
    mean_trajectory_rewards.append(mean_trajectory_reward_for_theta_dash)

  logging.info('coeffs= %s: ER_pos=%.4f, ER_neg=%.4f', coeffs,
               mean_trajectory_rewards[0], mean_trajectory_rewards[1])
  return results
