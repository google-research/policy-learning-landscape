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

"""Smoke tests for policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from eager_pg import policies

import tensorflow as tf

OBS_SPACE_SIZE = 10
ACT_SPACE_SIZE = 3
BATCH_SIZE = 32


def fake_observations(batch_size):
  return tf.constant(
      np.random.normal(size=(batch_size, OBS_SPACE_SIZE)), dtype=tf.float32)


class PoliciesTest(tf.test.TestCase, parameterized.TestCase):
  """Common class to run tests for policies."""

  @parameterized.named_parameters(
      dict(
          testcase_name='NormalFixedStd',
          policy_fn=policies.NormalPolicyFixedStd,
          policy_fn_arguments={'std': 0.5},
          policy_type='continuous'),
      dict(
          testcase_name='NormalLearnedStd',
          policy_fn=policies.NormalPolicyLearnedStd,
          policy_fn_arguments={'init_std': 0.5},
          policy_type='continuous'),
      dict(
          testcase_name='Categorical',
          policy_fn=policies.CategoricalPolicy,
          policy_fn_arguments={},
          policy_type='discrete'))
  def test_policy(self, policy_fn, policy_fn_arguments, policy_type):
    """Test if policy return types are good."""

    # Make a policy.
    policy = policy_fn(ACT_SPACE_SIZE, **policy_fn_arguments)

    # Pass in some data to see if things work.
    obs = fake_observations(BATCH_SIZE)
    data = policy(obs)

    # Smoke tests.
    self.assertIsNotNone(data)
    self.assertEqual(len(data), 3)

    action, log_prob, entropy = data

    # Shape checks.
    self.assertEqual(log_prob.shape, (BATCH_SIZE,))
    self.assertEqual(entropy.shape, (BATCH_SIZE,))

    if policy_type == 'discrete':
      self.assertEqual(action.shape, (BATCH_SIZE,))
    else:
      self.assertEqual(action.shape, (BATCH_SIZE, ACT_SPACE_SIZE))

    # Test get_dist() methods.
    self.assertIsInstance(policy.get_dist(obs), tf.distributions.Distribution)

    policy_copy = policy.get_deterministic_copy()
    # Ensure that the new policy is deterministic but the original is unchanged.
    if isinstance(policy_copy, policies.CONTINUOUS):
      self.assertAlmostEqual(policy_copy.std.numpy(), 0.0)
      self.assertAlmostEqual(policy.std.numpy(), 0.5)
    elif isinstance(policy_copy, policies.DISCRETE):
      self.assertAlmostEqual(policy_copy._softmax_temperature.numpy(), 1e-6)
      self.assertAlmostEqual(policy._softmax_temperature.numpy(), 1.0)

  def test_normal_policies_learnable_variables(self):
    """Make sure that fixed_std will not learn the standard deviation."""
    learned_std = policies.NormalPolicyLearnedStd(ACT_SPACE_SIZE, init_std=0.5)
    fixed_std = policies.NormalPolicyFixedStd(ACT_SPACE_SIZE, std=0.5)

    # Dummy initialize.
    learned_std(fake_observations(BATCH_SIZE))
    fixed_std(fake_observations(BATCH_SIZE))

    # Make sure standard deviations exist.
    self.assertIsNotNone(learned_std.std)
    self.assertIsNotNone(fixed_std.std)

    # Ensure that std is not trainable for the fixed_std case.
    self.assertEqual(len(learned_std.trainable_variables), 3)
    self.assertEqual(len(fixed_std.trainable_variables), 2)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
