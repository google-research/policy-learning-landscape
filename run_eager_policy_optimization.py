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

r"""Launcher script for optimizing policies via REINFORCE.

A multipurpose script to optimize a policy in an environment.

To run cartpole simply do:

```
python3 run_eager_policy_optimization.py \
--env CartPole-v0 \
--policy_type discrete
```

To run something from Mujoco you _must_ have it installed. To run an environment
from there:
```
python3 run_eager_policy_optimization.py \
--env Hopper-v1 \
--policy_type normal \
--std 0.5
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import flags
from eager_pg import experiment_utils
from eager_pg.algorithms import reinforce
import tensorflow as tf

from tensorflow.contrib import eager as tfe


DECAY_STEPS = 100  # Number of steps after which to decay learning rate.


flags.DEFINE_integer('updates', 100, 'Number of updates for the optimizer.')
flags.DEFINE_float('std', 0.1, 'Standard deviation for the Normal policy.')
flags.DEFINE_string('env', 'CartPole-v0', 'Name of Environment.')
flags.DEFINE_string('optimizer', 'sgd', 'Optimizer to use.'
                    'Available options: {sgd, rmsp}.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning Rate for the optimizer.')
flags.DEFINE_float('decay_rate', -1, 'Decay rate for inverse time decay.')
flags.DEFINE_integer('env_seed', 0, 'Seed for the environment.')
flags.DEFINE_integer('optimizer_seed', 1, 'Seed for the optimizer.')
flags.DEFINE_integer('start_parameter_seed', 0,
                     'Seed for the parameter vector initializer.')
flags.DEFINE_string(
    'base_save_dir', None, 'Location where to save checkpoints and summaries.')
flags.DEFINE_integer('batch_size', 16,
                     'Number of environment processes to run in parallel.')
flags.DEFINE_integer(
    'max_steps_env', 100000, 'Maximum number of steps to run in '
    'the environment before termination.')
flags.DEFINE_string('policy_type', 'discrete',
                    'Type of policy. Available options: {discrete, normal}.')
flags.DEFINE_float('discount', 0.995, 'Discount factor.')
flags.DEFINE_integer('save_parameter_interval', 5000,
                     'Saves parameters after these many updates.')
flags.DEFINE_integer('checkpoint_interval', 1000,
                     'Saves checkpoints after these many updates.')
flags.DEFINE_integer('summary_interval', 1000,
                     'Saves summaries after these many updates.')
flags.DEFINE_integer('deterministic_eval_interval', 1000,
                     'Runs deterministic evaluation after these many updates.')
flags.mark_flag_as_required('base_save_dir')
FLAGS = flags.FLAGS
logging = tf.logging
gfile = tf.gfile


def main(_):
  tf.logging.set_verbosity(logging.INFO)
  tf.logging.info('Welcome to the training script!')

  save_dir = experiment_utils.build_save_dir(
      FLAGS.base_save_dir, FLAGS.env, FLAGS.std, FLAGS.optimizer_seed,
      FLAGS.decay_rate, FLAGS.learning_rate, FLAGS.batch_size)

  experiment_utils.set_random_seed(FLAGS.optimizer_seed)

  objective = experiment_utils.get_objective('REINFORCE')

  env = experiment_utils.get_batched_environment(FLAGS.env, FLAGS.batch_size)

  # Load the policy network using arguments specific for the type of policy.
  layer_arguments = experiment_utils.get_layer_arguments(
      FLAGS.optimizer_seed, FLAGS.start_parameter_seed)

  policy = experiment_utils.get_policy(
      FLAGS.policy_type, env, layer_arguments, FLAGS.std)

  # Use the tfe.Variable so that it can be `assign`ed later.
  learning_rate = tfe.Variable(FLAGS.learning_rate)

  learning_rate_decay_fn = experiment_utils.get_learning_rate_decay_fn(
      learning_rate, FLAGS.decay_rate, DECAY_STEPS)

  optimizer = experiment_utils.get_optimizer(FLAGS.optimizer, learning_rate)

  trainer = reinforce.REINFORCE(
      env,
      policy,
      objective,
      optimizer,
      discount=FLAGS.discount,
      max_steps=FLAGS.max_steps_env,
      learning_rate=learning_rate,
      learning_rate_decay=learning_rate_decay_fn,
      deterministic_eval_frequency=FLAGS.deterministic_eval_interval)

  # Logging information.
  trainer.create_checkpointer(
      save_dir, checkpoint_frequency=FLAGS.checkpoint_interval)
  trainer.create_parameter_saver(
      save_dir, parameter_frequency=FLAGS.save_parameter_interval)
  trainer.create_summary_writer(
      save_dir, summary_frequency=FLAGS.summary_interval)

  trainer.restore_checkpoint()  # Restore checkpoint if it exists.

  start_time = time.time()
  trainer.train(FLAGS.updates)
  end_time = time.time()
  tf.logging.info('Training complete. Took %.2f seconds for %d updates. '
                  '(%.4f seconds per update)',
                  end_time - start_time, FLAGS.updates,
                  (end_time - start_time) / FLAGS.updates)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.app.run(main)
