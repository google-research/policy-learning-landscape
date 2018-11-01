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
r"""A simple interpolation experiment between two parameter vectors.

Run a simple interpolation between two parameter vectors and plot it the
landscape between them. Note that the plotting functionality is pretty
rudimentary. What you probably want to do is refer to the colab notebooks.

"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

from absl import flags
import gym
import matplotlib.pyplot as plt
import numpy as np
from eager_pg import batch_env
from eager_pg import env_spec
import common_flags
import multiprocessing_tools
import tensorflow as tf


# pylint: disable=invalid-name
BatchEnv = batch_env.BatchEnv
# pylint: enable=invalid-name

FLAGS = flags.FLAGS
logging = tf.logging
gfile = tf.gfile

flags.adopt_module_key_flags(common_flags)

flags.DEFINE_string('file_name', 'interpolation.npy',
                    'Name of numpy file to save data into.')
flags.DEFINE_string('save_dir', None, 'Location where to save results.')
flags.DEFINE_string('fig_title', 'Interpolation', 'Figure title.')
flags.DEFINE_integer('config', None, 'A configuration value.')


def create_env():
  """Creates an environment to be used within a process."""
  env = BatchEnv([gym.make(FLAGS.env) \
                  for _ in range(FLAGS.batch_size)])
  spec = env_spec.EnvSpec(env)
  return env, spec, {
      'max_steps_env': FLAGS.max_steps_env,
      'n_trajectories': FLAGS.n_trajectories
  }


def interpolate(std):
  """Does an interpolation for a certain standard deviation."""

  with gfile.Open(FLAGS.p1, 'r') as file_:
    theta0 = np.load(file_, allow_pickle=False)
  with gfile.Open(FLAGS.p2, 'r') as file_:
    theta1 = np.load(file_, allow_pickle=False)

  def theta_creator(alpha):
    """Linear combination of the two alphas."""
    return theta0 * (1 - alpha) + alpha * theta1

  alphas = np.linspace(FLAGS.alpha_start, FLAGS.alpha_end, FLAGS.n_alphas)

  policy_arguments = {'std': std}
  prepared_arguments = []

  logging.info('Preparing arguments for interpolation with std %f.', std)
  for alpha in alphas:
    prepared_arguments.append([(alpha,), theta_creator, policy_arguments,
                               create_env])

  logging.info('Preparation complete.')

  # pylint: disable=line-too-long
  file_name = FLAGS.file_name.format(std, '{}')
  _, final_file_name = multiprocessing_tools.managed_multiprocessing_loop_to_numpy(
      prepared_arguments,
      save_dir=FLAGS.save_dir,
      file_name_skeleton=file_name,
      save_every=FLAGS.save_every)
  # pylint: enable=line-too-long

  return final_file_name


def plot_interpolation(file_names):
  """Plots the results for each standard deviation."""
  fig = plt.figure()
  ax = fig.add_subplot(111)

  # Generate the x-axis.
  alphas = np.linspace(FLAGS.alpha_start, FLAGS.alpha_end, FLAGS.n_alphas)

  # Generate the colors.
  colors = plt.cm.Dark2(np.linspace(0, 1, len(file_names)))

  # Loop over each std and plot it.
  for color, std in zip(colors, file_names.keys()):
    with gfile.Open(file_names[std], 'r') as f_:
      interpolation_grid = np.load(f_)
    logging.info('Loaded interpolation grid for %f from %s.', std,
                 file_names[std])
    ax.plot(alphas, interpolation_grid, color=color, label='std={}'.format(std))

  x_ticks = [0, 1]
  x_ticklabels = [FLAGS.title1, FLAGS.title2]
  if FLAGS.alpha_start < 0:
    x_ticks.insert(0, FLAGS.alpha_start)
    x_ticklabels.insert(0, str(FLAGS.alpha_start))
  if FLAGS.alpha_end > 1:
    x_ticks.append(FLAGS.alpha_end)
    x_ticklabels.append(str(FLAGS.alpha_end))

  ax.set_xticks(x_ticks)
  ax.set_xticklabels(x_ticklabels)
  ax.set_xlabel(r'$\alpha$')
  ax.set_ylabel('Value')
  ax.set_ylim([FLAGS.min_v, FLAGS.max_v])
  ax.legend(
      fancybox=True, framealpha=0.5, loc='lower right', fontsize='x-small')
  ax.set_title(FLAGS.fig_title)
  fig.tight_layout()

  logging.info('Plotting done.')

  plot_save_file = FLAGS.file_name.replace('std{}_', '')
  plot_save_file = plot_save_file.replace(
      '{}.npy',
      FLAGS.fig_title.replace(' ', '_') + '.png')
  plot_save_path = os.path.join(FLAGS.save_dir, plot_save_file)

  with gfile.Open(plot_save_path, 'w') as f_:
    fig.savefig(f_)

  logging.info('Saved plot.')


def main(_):
  logging.set_verbosity(logging.INFO)
  logging.info('Running 1D Interpolation Experiment!')

  # Checking if the flags are good.
  if FLAGS.stds is None:
    if FLAGS.config == 1:
      FLAGS.stds = [1.0, 5.0]
    elif FLAGS.config == 2:
      FLAGS.stds = [1.0, 10.0]
    elif FLAGS.config == 3:
      FLAGS.stds = [0, 1.0, 5.0, 10.0, 15.0]
    elif FLAGS.config == 4:
      FLAGS.stds = [0, 1.0, 5.0, 10.0, 100.0]
    elif FLAGS.config == 5:
      FLAGS.stds = [0, 0.05, 0.1, 0.5, 1.0, 2.0]
    else:
      raise ValueError('Unknown config. Supply FLAGS.stds instead!')

  if FLAGS.save_dir is None:
    raise ValueError('Please specify where you want to save data.')

  # Setup file names and make saving directories.
  FLAGS.file_name = 'std{}_' + FLAGS.fig_title + FLAGS.file_name.replace(
      '.npy', '{}.npy')
  gfile.MakeDirs(FLAGS.save_dir)
  logging.info('Interpolation will be saved to %s/%s', FLAGS.save_dir,
               FLAGS.file_name)

  # Set the random seeds.
  tf.set_random_seed(FLAGS.global_seed)
  np.random.seed(FLAGS.global_seed)
  random.seed(FLAGS.global_seed)

  # Loop over each standard deviation in the list and do the interpolation.
  # Save the file name for each interpolation x standard deviation combo.
  # Use these to cache partial results + plot them at the end.
  if not FLAGS.visualize_only:
    for std in FLAGS.stds:
      interpolate(std)

  # Plotting
  # Set the path suitable for globbing.
  FLAGS.precomputed_interpolation = os.path.join(
      FLAGS.save_dir, FLAGS.file_name.format('*', ''))

  file_names = {}

  if FLAGS.precomputed_interpolation is None:
    raise ValueError('You must provide --precomputed_interpolation if using '
                     '--visualize_only')
  else:
    # Glob file paths and standard deviations.
    raw_file_names = gfile.Glob(FLAGS.precomputed_interpolation)
    for file_name in raw_file_names:
      # Assuming file names are of the form 'std0.00_my_name_here.npy'
      # Doing the split and slicing out the first three characters
      # allows us to get the standard deviation.
      std = float(os.path.basename(file_name).split('_')[0][3:])
      file_names[std] = file_name

  # Now that we have a dictionary of file names we just need to plot those
  # on the same plot.
  plot_interpolation(file_names)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.app.run(main)
