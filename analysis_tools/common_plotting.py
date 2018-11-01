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

"""Common tools for plotting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import seaborn as sns


# Colorblind friendly colors thanks to Utku and the internet.
FRIENDLY_COLORS = ['#377eb8', '#f781bf', '#4daf4a', '#a65628', '#984ea3',
                   '#ff7f00', '#999999', '#e41a1c', '#dede00']
# Stds used in most of our experiments.
STDS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 0.01, 'default']
STD2COLOR = dict(zip(map(str, STDS), FRIENDLY_COLORS))


def get_colors(n_colors, palette='colorblind'):
  """Abstraction to ensure all our plots have consistent colors.

  Args:
    n_colors: The number of colors to return.
    palette: The palette to use. Recommended {Dark2, colorblind}

  Returns:
    A list of colors that can be used to plotting.
  """
  return sns.color_palette(palette, n_colors)


def get_standardized_color(std):
  """Returns a color based on a fixed color palette that maps std to color."""
  std = str(std)
  if std in STD2COLOR:
    return STD2COLOR[str(std)]
  else:
    return STD2COLOR['default']


def get_ax(ax, figsize=None):
  """Create or reuse axes if it exists.

  Use this when you want to dynamically handle creating a new axis if it doesn't
  exist or reuse an existing one. For example:

  ```
  def add_curve(c, ax=None):
    ax = get_ax(ax)
    ax.plot(np.random.normal(c, 1, size=(100, )))
    return ax

  ax = add_curve(c=4)
  ax = add_curve(c=100, ax=ax)
  ax = add_curve(c=200, ax=ax)
  ```

  Args:
    ax: A matplotlib axes or None. If None, a figure will be created.
    figsize: The size of the figure to create if an ax is not given.

  Returns:
    A matplotlib axes.
  """
  if ax is None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
  return ax


if __name__ == '__main__':
  pass
