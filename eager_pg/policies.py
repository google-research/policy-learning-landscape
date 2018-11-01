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

"""A few generic policies to use in a variety of tasks.

A few simple policies that are implemented as tf.keras.Model objects to make it
simple to use with Tensorflow Eager.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

from tensorflow.contrib import eager as tfe


class CategoricalPolicy(tf.keras.Model):
  """Categorical policy to pick actions from a discrete action set."""

  def __init__(self,
               action_space_size,
               layer_arguments=None,
               softmax_temperature=1.0):
    """Learnable categorical policy for agents to pick discrete actions.

    Args:
      action_space_size: An int representing the size of the action space.
      layer_arguments: A dictionary containing arguments to pass to the Dense
        layer.
      softmax_temperature: A float scaling for the softmax logits. This should
        be 1.0 for any reasonable use case. The only time it is reasonable to
        want to modify this is during evaluation, to make a policy approximatley
        deterministic.
    """
    if layer_arguments is None:
      layer_arguments = {}

    super(CategoricalPolicy, self).__init__()
    self.linear_layer = tf.keras.layers.Dense(action_space_size,
                                              **layer_arguments)
    self._softmax_temperature = tfe.Variable(
        name='softmax_temperature', dtype=tf.float32, trainable=False,
        initial_value=tf.constant(softmax_temperature, tf.float32))

  def set_softmax_temperature(self, new_temperature):
    """Sets the softmax scaling to a new value."""
    self._softmax_temperature.assign(
        tf.constant(new_temperature, dtype=tf.float32))

  def get_dist(self, observation):
    """Returns the action distribution given an observation.

    Args:
      observation: A batch of observations shaped (batch_size, obs_space)

    Returns:
      distribution: a tf.distributions.Categorical object that can be used
        for downstream sampling and manipulation.
    """
    logits = self.linear_layer(observation) / self._softmax_temperature
    dist = tf.distributions.Categorical(logits)
    return dist

  @tfe.defun
  def call(self, observation):
    """Returns action and associated information from the categorical policy.

    Args:
      observation: A batch of observations shaped (batch_size, obs_space).

    Returns:
      sampled_actions: A batch of actions shaped (batch_size, action_space).
      action_log_probs: The log probability of each action shaped (batch_size,).
      entropy: The entropy of each action shaped (batch_size,).
    """
    dist = self.get_dist(observation)
    sampled_action = dist.sample()
    action_log_prob = dist.log_prob(tf.stop_gradient(sampled_action))
    entropy = dist.entropy()
    return sampled_action, action_log_prob, entropy

  def get_deterministic_copy(self):
    """Returns a deterministic version of this policy."""
    policy_copy = copy.deepcopy(self)
    approximately_deterministic_scaling = 1e-6
    policy_copy.set_softmax_temperature(approximately_deterministic_scaling)
    return policy_copy


class NormalPolicyFixedStd(tf.keras.Model):
  """Normal policy with a fixed standard deviation."""

  def __init__(self,
               action_space_size,
               std,
               layer_arguments=None):
    """Normal Policy with a fixed standard devation for continuous actions.

    Args:
      action_space_size: The size of the action space.
      std: The standard deviation for the sampling distribution.
      layer_arguments: Arguments to pass to the Dense layer that calculates the
        mean of the distribution.
    """
    super(NormalPolicyFixedStd, self).__init__()

    if layer_arguments is None:
      layer_arguments = {}

    self.linear_layer = tf.keras.layers.Dense(
        action_space_size, name='mu', **layer_arguments)

    self._std = tfe.Variable(name='std',
                             dtype=tf.float32,
                             trainable=False,
                             initial_value=tf.constant(std, tf.float32))

  def set_std(self, std):
    """Sets the standard deviation."""
    self._std.assign(tf.constant(std, dtype=tf.float32))

  @property
  def std(self):
    """Get the standard deviation of this policy."""
    return self._std

  def get_dist(self, observation):
    """Returns the action distribution given an observation.

    Args:
      observation: A batch of observations shaped (batch_size, obs_space).
    Returns:
      distribution: a tf.distributions.Normal object that can be used
        for downstream sampling and manipulation.
    """
    mu = self.linear_layer(observation)
    dist = tf.distributions.Normal(mu, self.std)
    return dist

  @tfe.defun
  def call(self, observation):
    """Execute the policy at an observation.

    Args:
      observation: A batch of observations shaped (batch_size, obs_space).
    Returns:
      sampled_actions: A batch of actions shaped (batch_size, action_space).
      action_log_probs: The log probability of each action shaped (batch_size,).
      entropy: The entropy of each action shaped (batch_size,).
    """
    dist = self.get_dist(observation)
    sampled_action = dist.sample()
    action_log_prob = dist.log_prob(tf.stop_gradient(sampled_action))
    entropy = dist.entropy()
    return (sampled_action, tf.reduce_sum(action_log_prob, -1),
            tf.reduce_sum(entropy, -1))

  def get_deterministic_copy(self):
    """Returns a deterministic version of this policy."""
    policy_copy = copy.deepcopy(self)
    policy_copy.set_std(0.0)
    return policy_copy


class NormalPolicyLearnedStd(NormalPolicyFixedStd):
  """Normal policy where the standard deviation is learned."""

  def __init__(self, action_space_size, init_std, layer_arguments=None):
    """Normal Policy with learned standard deivation for continuous actions.

    Args:
      action_space_size: The size of the action space.
      init_std: The initial standard deviation to use for the sampling
        distribution. This is learnable.
      layer_arguments: Arguments to pass to the Dense layer that calculates the
        mean of the distribution.
    """
    super(NormalPolicyLearnedStd, self).__init__(
        action_space_size, init_std, layer_arguments=layer_arguments)

    self.log_std = tfe.Variable(
        name='log_std',
        dtype=tf.float32,
        trainable=True,
        initial_value=tf.constant(tf.log(init_std), tf.float32))

  def set_std(self, std):
    """Reset the standard deviation of the normal policy."""
    self.log_std.assign(tf.log(tf.constant(std, tf.float32)))

  @property
  def std(self):
    """Returns the standard deviation."""
    return tf.exp(self.log_std)

CONTINUOUS = (NormalPolicyFixedStd, NormalPolicyLearnedStd)
DISCRETE = (CategoricalPolicy,)

if __name__ == '__main__':
  pass
