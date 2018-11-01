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

"""Common flags for interpolation experiments.

Modules that import these flags can choose to use any number of these flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS

# Location of the parameter vectors. Upto three are supported for now.
flags.DEFINE_string('p1', None, 'Location of the first parameter vector.')
flags.DEFINE_string('p2', None, 'Location of the second parameter vector.')
flags.DEFINE_string('p3', None, 'Location of the third parameter vector.')

## Coefficient information for interpolation.
# The first coefficient is called alpha.
flags.DEFINE_float('alpha_start', 0, 'Start alpha.')
flags.DEFINE_float('alpha_end', 1, 'End alpha.')
flags.DEFINE_integer('n_alphas', 10, 'Number of alphas.')

# The second coefficient is called beta.
flags.DEFINE_float('beta_start', 0, 'Start beta.')
flags.DEFINE_float('beta_end', 1, 'End beta.')
flags.DEFINE_integer('n_betas', 10, 'Number of betas.')

## Plotting details
# The limits for the plots.
flags.DEFINE_float('max_v', 2500, 'Maximum value on the y/z-axis.')
flags.DEFINE_float('min_v', 0, 'Minimum value on the y/z-axis.')

# The title for each parameter vector passed above.
flags.DEFINE_string('title1', 'p1', 'Title for p1.')
flags.DEFINE_string('title2', 'p2', 'Title for p2.')
flags.DEFINE_string('title3', 'p3', 'Title for p3.')

# Allows skipping of interpolating just to do plotting.
flags.DEFINE_bool(
    'visualize_only', False, 'Will only do visualization. '
    'If passed must specify --precomputed_interpolation')
flags.DEFINE_string('precomputed_interpolation', None,
                    'File path to a precomputed interpolation.')

# Policy and environment details.
flags.DEFINE_string('env', 'Hopper-v1', 'Name of Environment.')
flags.DEFINE_integer('env_seed', 0, 'Seed for the environment.')
flags.DEFINE_integer('global_seed', 1, 'Seed for all the rngs.')
flags.DEFINE_integer('save_every', 10, 'Save results after these many epochs.')
flags.DEFINE_integer('batch_size', 16, 'Number of environments to run.')
flags.DEFINE_integer('n_trajectories', 128, 'Number of trajectories to use.')
flags.DEFINE_integer(
    'max_steps_env', 1500, 'Maximum number of steps to run in the environment '
    'before termination.')
flags.DEFINE_float('std', None, 'Standard deviations for the policy.')
flags.DEFINE_multi_float('stds', None,
                         'A list of standard deviations for the policy.')
flags.DEFINE_string('policy_type', 'normal',
                    'Type of policy. Either `discrete` or `normal`.')

