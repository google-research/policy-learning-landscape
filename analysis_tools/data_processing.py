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

"""Tools to prepare learning curves for plotting.

Some utilities include:
- Sanitizing the number of environment steps to ensure that they are always
increasing.
- Applying linear smoothing, similar to that in Tensorboard.
- Producing an averaged curve with error bars from multiple replicates of an
experiment. These should be stored as: `path/to/experiment/SeedX` where X is
a number. Each folder should have tensorboard summaries that can be read.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import data_io
from scipy import interpolate


DEFAULT_SMOOTHING_WEIGHT = 0.9


def sanitize_env_steps(env_steps_sequence):
  """This function ensures that the env_steps_sequence is always increasing.

  Sometimes env_steps can reset to zero (when badly checkpointed). This function
  will ensure that the computations are correct by just adding to the last
  maximum count found. For example, consider the badly checkpointed sequence:
  ```
  bad_sequence = [5, 10, 15, 5, 10, 15]
  ```
  At index 3 the job preempted and restarted. The answer should be:
  ```
  expected_answer = [5, 10, 15, 20, 25, 35]
  ```
  This function will return a result such that:
  ```
  assert all(a == b for a,b in zip(sanitize_env_steps(bad_sequence),
                                   expected_answer))
  ```

  Args:
    env_steps_sequence: A list of floats representing environment steps taken.

  Returns:
    A list of floats representing the environment steps taken sanitized for
    issues in checkpointing.
  """
  logging.log_first_n(logging.INFO,
                      'Sanitizing data. This will show only once.', 1)
  sanitized_env_steps = []
  last_max = 0
  for i in range(len(env_steps_sequence) - 1):
    xt, xtp = env_steps_sequence[i], env_steps_sequence[i + 1]
    sanitized_env_steps.append(xt + last_max)
    if xtp < xt:
      # Reset occurred between t and t+1.
      # Set the last_max to be the current step so that we continue by just
      # adding to the current step.
      last_max = xt
  sanitized_env_steps.append(xtp + last_max)
  return np.array(sanitized_env_steps)


def apply_linear_smoothing(data, smoothing_weight=DEFAULT_SMOOTHING_WEIGHT):
  """Smooth curves using a exponential linear weight.

  This smoothing algorithm is the same as the one used in tensorboard.

  Args:
    data: The iterable containing the data to smooth.
    smoothing_weight: A float representing the weight to place on the moving
      average.

  Returns:
    A list containing the smoothed data.
  """
  assert len(data), 'No curve to smooth.'  # pylint: disable=g-explicit-length-test,line-too-long
  if smoothing_weight <= 0:
    return data
  last = data[0]
  smooth_data = []
  for x in data:
    if not np.isfinite(last):
      smooth_data.append(x)
    else:
      smooth_data.append(last * smoothing_weight + (1 - smoothing_weight) * x)
    last = smooth_data[-1]
  return smooth_data


def extract_average_curve(file_path,
                          n_points=500,
                          y_tag='mean_deterministic_trajectory_reward',
                          x_tag=None,
                          min_trajectory_len=0,
                          smoothing_weight=DEFAULT_SMOOTHING_WEIGHT,
                          skip_seed_path_glob=False):
  """Extract a curve averaged over all experimental replicates.

  Args:
    file_path: The path to where the experimental replicates are saved.
    n_points: The number of points to plot.
    y_tag: A string representing the data that will be plotted on the y-axis.
    x_tag: A string representing the data that will be plotted on the x-axis.
      This should be None or `env_steps_at_deterministic_eval` to be something
      meaningful.
    min_trajectory_len: The minimum number of elements in the optimization
      trajectory or the minimum number of environment steps after which to
      consider curves to plot. If this is 0 it will truncate all replicates to
      the shortest replicate.
    smoothing_weight: A float representing how much smoothing to do to the data.
    skip_seed_path_glob: A boolean indicating if glob should be skipped. If it
      is skipped then `file_path` should directly lead to the directory with all
      the data.

  Returns:
    A tuple containing three numpy arrays representing:
      - x: The values to plot on the x-axis.
      - y_mean: The values of plot on the y-axis.
      - y_std: The standard deviation/spread on the y-axis.
  """

  if skip_seed_path_glob:
    seed_paths = [file_path]
  else:
    seed_paths = data_io.get_replicates(file_path)

  # Store the minimum starting and maximum ending values to do truncation.
  minimum_x = []
  maximum_x = []
  interpolators = []
  for seed_path in seed_paths:
    ea = data_io.load_events(seed_path)
    x, y = data_io.extract_np_from_scalar_events(ea.Scalars(y_tag))

    if x_tag is not None:
      _, x = data_io.extract_np_from_scalar_events(ea.Scalars(x_tag))

    min_x = np.min(x)
    max_x = np.max(x)
    if min_trajectory_len > max_x:
      logging.info('Skipping: %f', min_x)
      continue
    interpolator = interpolate.interp1d(x, y)
    interpolators.append(interpolator)
    minimum_x.append(min_x)
    maximum_x.append(max_x)

  logging.info('%s, \n minimum_x: %s, maximum_x: %s',
               file_path, minimum_x, maximum_x)

  start_point = np.max(minimum_x)
  end_point = np.min(maximum_x)
  x = np.linspace(start_point, end_point, n_points)
  ys = []
  for interpolator in interpolators:
    y = interpolator(x)
    if smoothing_weight is not None:
      y = apply_linear_smoothing(y, smoothing_weight=smoothing_weight)
    ys.append(y)

  y_stacked = np.stack(ys)
  y_mean = np.mean(y_stacked, 0)
  y_std = np.std(y_stacked, 0)

  return x, y_mean, y_std


if __name__ == '__main__':
  pass
