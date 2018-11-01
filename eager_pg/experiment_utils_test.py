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

"""Unit and integration tests for experiment_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from eager_pg import env_spec
from eager_pg import experiment_utils
from eager_pg import rbf_env_spec
from eager_pg.algorithms import reinforce
import tensorflow as tf

from tensorflow.contrib import eager as tfe

ENV_NAME = 'Pendulum-v0'
BATCH_SIZE = 5
BASE_SAVE_DIR = '/tmp/integration_test'
STD = 1.0
OPTIMIZER_NAME = 'sgd'
LEARNING_RATE = 0.05
OPTIMIZER_SEED = 99
DECAY_RATE = -1.0
DECAY_STEPS = 100


class ExperimentUtilsTest(tf.test.TestCase):

  def test_build_save_dir(self):
    """Test if the built saving directory is correct."""
    save_dir = experiment_utils.build_save_dir(BASE_SAVE_DIR, ENV_NAME, STD,
                                               OPTIMIZER_SEED, DECAY_RATE,
                                               LEARNING_RATE, BATCH_SIZE)

    self.assertEqual(
        save_dir, '/tmp/integration_test/ObjREINFORCE/'
        'EnvPendulum-v0/BS5/LR0.05/Std1.0/Seed99/')

  def test_integration(self):
    """Test if components work together."""
    experiment_utils.set_random_seed(OPTIMIZER_SEED)
    objective = experiment_utils.get_objective('REINFORCE')
    env = experiment_utils.get_batched_environment(ENV_NAME, BATCH_SIZE)
    layer_arguments = {}
    policy = experiment_utils.get_policy('normal', env, layer_arguments, STD)

    learning_rate = tfe.Variable(LEARNING_RATE)

    learning_rate_decay_fn = experiment_utils.get_learning_rate_decay_fn(
        learning_rate, DECAY_RATE, DECAY_STEPS)

    optimizer = experiment_utils.get_optimizer(OPTIMIZER_NAME, learning_rate)

    trainer = reinforce.REINFORCE(
        env,
        policy,
        objective,
        optimizer,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay_fn)

    # Logging information.
    save_dir = '/tmp/integration_test/'
    trainer.create_checkpointer(save_dir)
    trainer.create_parameter_saver(save_dir)
    trainer.create_summary_writer(save_dir)

    # Check if everything works well together.
    trainer.train(2)

  def test_env_spec_builder(self):
    """Test get_env_spec_builder returns correct env_specs and raises errors."""
    env = experiment_utils.get_batched_environment(ENV_NAME, BATCH_SIZE)
    self.assertIsInstance(
        experiment_utils.get_env_spec_builder()(env),
        env_spec.EnvSpec)

    self.assertIsInstance(
        experiment_utils.get_env_spec_builder(use_gpu=True)(env),
        env_spec.GPUEnvSpec)

    self.assertIsInstance(
        experiment_utils.get_env_spec_builder(featurizer='rbf')(env),
        rbf_env_spec.RBFEnvSpec)

    with self.assertRaises(NotImplementedError):
      experiment_utils.get_env_spec_builder(featurizer='rbf', use_gpu=True)

    with self.assertRaises(ValueError):
      experiment_utils.get_env_spec_builder(featurizer='nofeature!')


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
