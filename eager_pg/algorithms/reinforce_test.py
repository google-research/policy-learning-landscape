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

"""Tests for the REINFORCE algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from eager_pg import experiment_utils
from eager_pg import policies
from eager_pg.algorithms import reinforce
import tensorflow as tf
from tensorflow.contrib import eager as tfe


class REINFORCETest(tf.test.TestCase, parameterized.TestCase):
  """Common class to run tests for reinforce algorithm."""

  @parameterized.named_parameters(
      dict(
          testcase_name='Categorical',
          policy_name='discrete',
          policy_fn=policies.CategoricalPolicy,
          policy_fn_arguments={},
          env_name='CartPole-v0',
          learning_rate=0.01))
  def test_training(self, policy_name, policy_fn, policy_fn_arguments, env_name,
                    learning_rate):
    """Test that REINFORCE can train in different environments."""
    experiment_utils.set_random_seed(0)
    objective = experiment_utils.get_objective('REINFORCE')
    env = experiment_utils.get_batched_environment(env_name, 16)
    std = policy_fn_arguments['std'] if policy_name == 'normal' else None
    policy = experiment_utils.get_policy(policy_name, env, {}, std)

    learning_rate = tfe.Variable(learning_rate, trainable=False)

    optimizer = experiment_utils.get_optimizer('sgd', learning_rate)

    trainer = reinforce.REINFORCE(
        env,
        policy,
        objective,
        optimizer,
        max_steps=500,
        learning_rate=learning_rate)

    # Logging information.
    save_dir = '/tmp/reinforce_integration_test_{}/'.format(policy_name)
    trainer.create_checkpointer(save_dir)
    trainer.create_parameter_saver(save_dir)
    trainer.create_summary_writer(save_dir)

    det_stats_before_training = trainer.evaluate_deterministic_policy()
    stoch_stats_before_training = trainer.evaluate_policy()
    trainer.train(50)
    det_stats_after_training = trainer.evaluate_deterministic_policy()
    stoch_stats_after_training = trainer.evaluate_policy()
    # pylint: disable=line-too-long
    if isinstance(policy, policies.CONTINUOUS):
      self.assertGreater(
          det_stats_after_training['mean_deterministic_trajectory_reward'].numpy(),
          det_stats_before_training['mean_deterministic_trajectory_reward'].numpy())
    # pylint: enable=line-too-long
    self.assertGreater(
        stoch_stats_after_training['mean_trajectory_reward'].numpy(),
        stoch_stats_before_training['mean_trajectory_reward'].numpy())


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
