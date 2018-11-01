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

"""Utilities for running experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import random

from absl import flags
from absl import logging
import gym
import numpy as np
from eager_pg import batch_env
from eager_pg import env_spec
from eager_pg import objectives
from eager_pg import policies
from eager_pg import rbf_env_spec
import tensorflow as tf


FLAGS = flags.FLAGS
SAVE_DIR_TEMPLATE = 'Obj{objective}/Env{env}/BS{batch_size}/LR{learning_rate}/Std{std}/Seed{seed}/'  # pylint: disable=line-too-long


def get_batched_environment(env_name, batch_size):
  """Returns a batched version of the environment."""
  return batch_env.BatchEnv([gym.make(env_name) \
                             for _ in range(batch_size)])


def set_random_seed(seed):
  """Set all the random seeds.

  Args:
    seed: The seed to set for tensorflow, numpy and python.
  """
  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)


def build_save_dir(base_save_dir,
                   env_name,
                   std,
                   optimizer_seed,
                   decay_rate,
                   learning_rate,
                   batch_size,
                   objective='REINFORCE',
                   save_dir_template=SAVE_DIR_TEMPLATE):
  """Create the directory where output data will be saved.

  Args:
    base_save_dir: The base save directory (string).
    env_name: The name of the environment (string).
    std: The standard deviation for the policies (float).
    optimizer_seed: Seed used during optimization (int).
    decay_rate: The decay rate for the optimizer (float).
    learning_rate: The (initial) learning rate for the optimizer (float).
    batch_size: The number of trajectories used per update step (int).
    objective: Name of the objective for optimization (string).
    save_dir_template: The template saving directory (string).

  Returns:
    The constructed directory path where data should be saved.
  """
  constructed_dir = os.path.join(base_save_dir, save_dir_template)
  save_dir = constructed_dir.format(
      env=env_name,
      seed=optimizer_seed,
      std=std,
      decay=decay_rate,
      objective=objective,
      learning_rate=learning_rate,
      batch_size=batch_size)

  return save_dir


def get_layer_arguments(optimizer_seed,
                        start_parameter_seed=0,
                        nondeterministic_start=True):
  """Get arguments to be passed into constructors for layers in the policies.

  Args:
    optimizer_seed: Int seed for the optimizer.
    start_parameter_seed: Int seed for generating the starting parameter vector.
    nondeterministic_start: Boolean to determine if starting parameters should
      be non-deterministic. Will default to NOT having a fixed starting
      parameter vector.

  Returns:
    A dictionary containing the arguments to pass to the learnable part of
    a policy.
  """
  if nondeterministic_start:
    return {}
  else:
    # Keras' default Dense initializer is `glorot_uniform` so wrap that.
    def _deterministic_initializer(shape, **kwargs):
      tf.set_random_seed(start_parameter_seed)
      result = tf.keras.initializers.glorot_uniform(shape=shape, **kwargs)
      tf.set_random_seed(optimizer_seed)
      return result

    return {'kernel_initializer': _deterministic_initializer}


def get_policy(policy_name, env, layer_arguments, std):
  """Return the policy to be used in this environment.

  Args:
    policy_name: The name of the policy to use.
      Currently supported {'normal', 'discrete'}.
    env: The environment that this policy will be executed in.
    layer_arguments: The arguments to be passed to the constructor
      for learnable part of the policy.
    std: The standard deviation for a continuous policy.

  Raises:
    ValueError: If unknown policy is given.

  Returns:
    A keras object that represents the policy.
  """
  if policy_name == 'normal':
    policy = policies.NormalPolicyFixedStd(
        env.action_space.shape[0], std, layer_arguments=layer_arguments)
  elif policy_name == 'discrete':
    policy = policies.CategoricalPolicy(
        env.action_space.n, layer_arguments=layer_arguments)
  else:
    raise ValueError('Unknown policy provided.')
  return policy


def get_optimizer(optimizer_name, learning_rate):
  """Obtain the optimizer to use.

  Args:
    optimizer_name: Name of the optimizer to use.
    learning_rate: The initial learning rate to use, should be a float or a
      tfe.Variable

  Raises:
    ValueError: if an unknown optimizer_name is given.

  Returns:
    A tf.train.Optimizer object.
  """
  if optimizer_name == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer_name == 'rmsp':
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
  else:
    raise ValueError('Unknown optimizer selected.')

  return optimizer


def get_learning_rate_decay_fn(learning_rate, decay_rate, decay_steps):
  """Gets the learning rate decay function or None.

  See
  https://www.tensorflow.org/api_guides/python/train#Decaying_the_learning_rate
  for more details about how to use decay_rate and decay_steps.

  Args:
    learning_rate: A tfe.Variable with the learning rate.
    decay_rate: The magnitude of how much to decay as a float.
    decay_steps: The number of steps over which to decay as an int.

  Returns:
    learning_rate_decay_fn: Either None (no decay) or a function that can be
      executed to return the new learning rate after a certain step.
  """
  if decay_rate < 0:
    tf.logging.info('No decay used since it is less than 0.')
    learning_rate_decay_fn = None
  else:
    learning_rate_decay_fn = tf.train.polynomial_decay(
        learning_rate, tf.train.get_or_create_global_step(),
        decay_steps=decay_steps,
        end_learning_rate=learning_rate.numpy()*decay_rate)
    return learning_rate_decay_fn


def get_objective(objective_name='REINFORCE'):
  """Get the objective to optimize.

  Args:
    objective_name: Name of the objective.

  Raises:
    ValueError: When an unknown objective is given.

  Returns:
    The objective object to be used to compute the gradients with.
  """
  if objective_name == 'REINFORCE':
    return objectives.REINFORCE()
  else:
    raise ValueError('Unknown objective_name.')


def get_env_spec_builder(featurizer='none', use_gpu=False,
                         rbf_featurizer_kwargs=None):
  """Get a environment spec builder function.

  Args:
    featurizer: The featurizer to use as a string. Supports {'none', 'rbf'}.
    use_gpu: A boolean indicating if the GPU should be used.
    rbf_featurizer_kwargs: A dictionary containing keyword arguments for
      rbf_env_spec.RBFEnvSpec builder.

  Raises:
    NotImplementedError: If requesting gpu version of an env_spec that doesn't
      support it.
    ValueError: If requesting a featurizer that is not supported.
  Returns:
    An `env_spec.EnvSpec` class method.
  """
  if featurizer == 'rbf':
    if rbf_featurizer_kwargs is None:
      rbf_featurizer_kwargs = {'sample_mode': 'rollouts'}
    if use_gpu:
      raise NotImplementedError('rbf featurizer has no gpu implementation yet.')
    return functools.partial(rbf_env_spec.RBFEnvSpec, **rbf_featurizer_kwargs)
  elif featurizer == 'none':
    if use_gpu:
      logging.warning('EXPERIMENTAL: Using gpu environment!')
      return env_spec.GPUEnvSpec
    else:
      return env_spec.EnvSpec
  else:
    raise ValueError('Unknown featurizer provided. '
                     'Only support `none` and `rbf`.')


if __name__ == '__main__':
  pass
