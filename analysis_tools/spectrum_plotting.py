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

"""Library to obtain curvature and gradient spectra sampled loss functions.

This code extends the random sampling experiment. We can show that if we assume
that our loss function was quadratic: `f(x) = a^T x + x^T H x` then a projection
onto the x=y and x=-y axes returns the curvature (x^THx) and gradient (a^Tx)
information respectively. The methods in this file help us extract this
information from the sampled loss function.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import data_io
import scatter_plotting

# These axes recover the gradient and curvature spectrum when projecting the 2D
# scatter plot evaluations.
CURVATURE_AX = np.array([1, 1])  # x = y
GRADIENT_AX = np.array([1, -1])  # x = -y


def scalar_project(x, v):
  """Calculate the scalar projection of vector x onto vector v."""
  v_hat = v / np.linalg.norm(v)
  return np.dot(v_hat, x)


def get_gradient_projection(values_centered):
  """Project 2D points onto the x=-y axis which gives gradient information."""
  return np.apply_along_axis(
      lambda x: scalar_project(x, GRADIENT_AX), 1, values_centered)


def get_curvature_projection(values_centered):
  """Project 2D points onto the x=y axis which gives curvature information."""
  return np.apply_along_axis(
      lambda x: scalar_project(x, CURVATURE_AX), 1, values_centered)


def get_spectrum_data(file_path, std, alpha,
                      tag='mean_trajectory_return',
                      file_template=scatter_plotting.DEFAULT_FILE_NAME,
                      curvature=True, gradient=True,
                      data_extractor=data_io.extract_pairs,
                      file_reader=data_io.read_ndjson_file):
  """Get data necessary to calculate the curvature and gradient spectra.

  Args:
    file_path: The string path to where the files are saved.
    std: A string or float representing the standard deviation to extract
      information from.
    alpha: The distance travelled along the direction vectors.
    tag: The string representing the tag to extract from the data files.
    file_template: The template representing how the files are named.
    curvature: A boolean indicating that the curvature data will be returned.
    gradient: A boolean indicating that the gradient data will be returned.
    data_extractor: The data extraction helper object.
    file_reader: The file reading helper object.

  Returns:
    A dict containing:
      - noise_std: The standard deviation in the sampling noise.
      - gradient_spectrum_data: Data to calculate the gradient spectrum.
      - curvature_spectrum_data: Data to calculate the curvature spectrum.
  """

  full_file_path = os.path.join(file_path, file_template)

  (data_centered, _, noise_std, _) = scatter_plotting.extract_centered_data(
      std, alpha, tag, file_template=full_file_path,
      data_extractor=data_extractor, file_reader=file_reader)

  to_return = {'noise_std': noise_std}
  if gradient:
    to_return['gradient_spectrum_data'] = get_gradient_projection(data_centered)
  if curvature:
    to_return['curvature_spectrum_data'] = get_curvature_projection(data_centered)  # pylint: disable=line-too-long

  return to_return


def get_stats_from_histogram(counts, bins, noise_std, num_stds=2):
  r"""Get statistics from the histogram.

  Given the bin edges of a histogram and the counts in each bin, this function
  returns the proportion of values (i.e. sum of counts) that are above, below or
  inbetween the tolerance where `tolerance = init_std * num_stds`.

  Args:
    counts: A list containing the counts for each bin in the histogram.
      For example: `[4, 5]` represents that the first bin has 4 counts and the
      second bin has 5 counts.
    bins: A list containing the corners of a bin in a histogram.
      For example: `[0, 4, 8]` represents two bins: 0-4 and 4-8 in the
      histogram.
    noise_std: The standard deviation in the sampling error.
    num_stds: The number of standard deviations to allow tolerance.

  Returns:
    Proportion of values above the tolerance.
    Proportion of values below the tolerance.
  """
  tolerance = num_stds * noise_std
  total_count = np.sum(counts)
  pos_counts = np.sum(counts[np.where(bins[:-1] > tolerance)])
  prop_pos = pos_counts / total_count
  neg_counts = np.sum(counts[np.where(bins[:-1] < -tolerance)])
  prop_neg = neg_counts / total_count

  return prop_pos, prop_neg


if __name__ == '__main__':
  pass
