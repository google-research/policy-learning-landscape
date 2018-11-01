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

# pylint: disable=line-too-long
r"""Get the distribution of rewards for the paried random directions.

Given a parameter vector returns the change in reward for a number of random
directions around it bounded by an ball of size alpha. The main innovation in
this experiment is that the directions that are evaluated are paired so that we
can evaluate both for (alpha, -alpha) which allows us to detect more information
about the loss function.


After running `run_eager_policy_optimization.py` use this command to learn about
the optimization landscape around a parameter vector. Use as:

```
python paired_random_directions_experiment.py --p1 ./path/to/parameter/1/npy \
--save_dir ./path/to/save/in/ \
--alpha 0.5 --std 0.5 --n_directions 500
```

Data is saved into an ndjson file so technically speaking you can run reduce
`n_directions` to `100` and repeat this code 5 times to get similar sampling
results.

"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time

from absl import flags
import gym
import numpy as np
from eager_pg import batch_env
from eager_pg import env_spec
import common_flags
import landscape_explorers
import multiprocessing_tools
from six.moves import xrange
import tensorflow as tf


# pylint: disable=invalid-name
BatchEnv = batch_env.BatchEnv
# pylint: enable=invalid-name

FLAGS = flags.FLAGS
logging = tf.logging
gfile = tf.gfile

flags.adopt_module_key_flags(common_flags)

flags.DEFINE_string('file_name', 'paired_random_projections.npy',
                    'Name of numpy file to save data into.')
flags.DEFINE_string('save_dir', None, 'Location where to save results.')
flags.DEFINE_float('alpha', 0.5, 'The size of the ball.')
flags.DEFINE_integer('n_directions', 500,
                     'The number of directions to consider.')


def create_env():
  """Creates an environment to be used within a process."""
  env = BatchEnv([gym.make(FLAGS.env) \
                  for _ in range(FLAGS.batch_size)])
  spec = env_spec.EnvSpec(env)
  return env, spec, {
      'max_steps_env': FLAGS.max_steps_env,
      'n_trajectories': FLAGS.n_trajectories
  }


def random_direction_experiment():
  """Sets up arguments and runs the random directions experiment."""

  with gfile.Open(FLAGS.p1, 'r') as file_:
    theta0 = np.load(file_, allow_pickle=False)

  def theta_creator(alpha, unit_random_direction):
    """Returns a theta in a random direction."""
    return theta0 + alpha * unit_random_direction

  policy_arguments = {'std': FLAGS.std}
  coeff = (FLAGS.alpha, theta0.shape[0])

  # The first argument contains the return at the initial point.
  multiprocessing_argument = [
      coeff, theta_creator, policy_arguments, create_env
  ]

  prepared_arguments = []
  # All subsequent arguments are with the specified alpha.
  # This allows us to calculate the delta change in reward when plotting.
  for _ in xrange(FLAGS.n_directions):
    prepared_arguments.append(multiprocessing_argument)

  logging.info('Preparation complete: %s.', prepared_arguments[:3])

  _, final_file_name = (
      multiprocessing_tools.managed_multiprocessing_loop_to_ndjson(
          prepared_arguments,
          function_to_execute=(
              landscape_explorers.paired_landscape_explorer_parallel),
          save_dir=FLAGS.save_dir,
          file_name_skeleton=FLAGS.file_name,
          save_every=FLAGS.save_every,
      ))
  return final_file_name


def main(_):
  # Welcome message.
  logging.set_verbosity(logging.INFO)
  logging.info('Running Paired Random Directions Experiment!')

  # Checking if the flags are good.
  if FLAGS.std is None:
    raise ValueError('FLAGS.std is required.')

  if FLAGS.save_dir is None:
    raise ValueError('FLAGS.save_dir is required.')

  # Setup file names.
  file_name_append = 'std{}_alpha{}_'.format(FLAGS.std, FLAGS.alpha)
  FLAGS.file_name = FLAGS.file_name.replace('.npy', '{}.npy')
  FLAGS.file_name = file_name_append + FLAGS.file_name

  # Create save directory.
  gfile.MakeDirs(FLAGS.save_dir)
  logging.info('Interpolation will be saved to %s/%s', FLAGS.save_dir,
               FLAGS.file_name)

  # FLAGS.global_seed does nothing here because we want randomness between runs:
  # Since we want to evaluate for random alphas, we can just put in time.time()
  # here. This allows us to be pre-empted and not save repeats of the same
  # random values into the ndjson file possibly skewing our analysis.
  time_seed = int(time.time())
  logging.info('Seed used: %d', time_seed)
  tf.set_random_seed(time_seed)
  np.random.seed(time_seed)
  random.seed(time_seed)

  random_direction_experiment()


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.app.run(main)
