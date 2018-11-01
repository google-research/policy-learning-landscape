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

"""Tests getting and setting parameters using eager_pg.param_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from eager_pg import param_utils
from eager_pg import policies
import tensorflow as tf

RANDOM_OBS = np.random.normal(size=(50, 5)).astype(np.float32)
ACT_SPACE = 4


class ParamUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='NormalFixedStd',
          policy_fn=policies.NormalPolicyFixedStd,
          policy_fn_arguments={'std': 0.5}),
      dict(
          testcase_name='NormalLearnedStd',
          policy_fn=policies.NormalPolicyLearnedStd,
          policy_fn_arguments={'init_std': 0.5}),
      dict(
          testcase_name='Categorical',
          policy_fn=policies.CategoricalPolicy,
          policy_fn_arguments={}))
  def test_get_and_set_parameters(self, policy_fn, policy_fn_arguments):
    """Test that we can successfully get/set parameters."""
    policy = policy_fn(ACT_SPACE, **policy_fn_arguments)
    policy(tf.constant(RANDOM_OBS))  # Initialize policy.

    # Get Parameters.
    original_params = param_utils.get_flat_params(policy.trainable_variables)
    artificial_params = np.ones_like(original_params)

    if isinstance(policy, policies.NormalPolicyLearnedStd):
      artificial_params[-1] = 100

    # Set Parameters.
    policy = param_utils.set_flat_params(policy, artificial_params)
    retrieved_params = param_utils.get_flat_params(policy.trainable_variables)

    # Conditionally handle if the last parameter was correctly assigned.
    if isinstance(policy, policies.NormalPolicyLearnedStd):
      self.assertEqual(retrieved_params[-1], 100.0)
      self.assertTrue(np.allclose(retrieved_params[:-1], 1))
    else:
      self.assertTrue(np.allclose(retrieved_params, 1))

    # Now reset Original Parameters.
    param_utils.set_flat_params(policy, original_params)
    retrieved_params = param_utils.get_flat_params(policy.trainable_variables)
    self.assertTrue(np.allclose(retrieved_params, original_params))


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
