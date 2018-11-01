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

r"""Simple tools to create scatter plots for visualizing the landscape.

Use case examples:

1. Scatter plot for successively increasing alphas:
```
path_to_experiments = '/PATH/TO/EXPERIMENTS/'
ax = scatter_plotting.plot_2d_visualization(path_to_experiments, stds='1.0',
      alphas=['0.01', '0.1'])
```

2. Scatter plot for successively increasing stds:
```
ax = scatter_plotting.plot_2d_visualization(path_to_experiments,
      stds=['1.0', '2.0],
      alphas='0.01')
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import numpy as np

import common_plotting
import data_io

DEFAULT_LEGEND_STYLE = dict(frameon=True, framealpha=1, fontsize='x-small')
DEFAULT_FILE_NAME = 'std{std}_alpha{alpha}_paired_random_projections.ndjson'
DEFAULT_NOISE_INDICATOR_LINEWIDTH = 1.5


def extract_centered_data(std, alpha, tag,
                          file_template=DEFAULT_FILE_NAME,
                          data_extractor=data_io.extract_pairs,
                          file_reader=data_io.read_ndjson_file):
  """Extract centered data from a file containing objective evaluations.

  Args:
    std: A string or float representing the standard deviation to extract.
    alpha: A string or float representing the distance travelled from init.
    tag: The tag to extract from the data.
    file_template: A string representing the file path to extract data from.
    data_extractor: The data extraction helper object.
    file_reader: The file reading helper object.

  Returns:
    The centered objective values after taking the step and objective values
    for not taking the step.
  """
  # Read the raw, uncentered data.
  data_raw = data_extractor(
      file_reader(file_template.format(std=std, alpha=alpha)), tag)

  # Read the data containing sampling noise.
  init_datas = data_extractor(
      file_reader(file_template.format(std=std, alpha='0.0')), tag)

  # Center the raw data.
  init_std = np.std(init_datas, 0)
  init_datas = np.mean(np.array(init_datas), 0)
  data = np.array(data_raw) - init_datas

  return data, init_datas, init_std, data_raw


def _get_file_template(path_to_experiments, file_name_template):
  """Get file template from where to load data."""
  return os.path.join(path_to_experiments, file_name_template)


def _plot_for_hyperparameter_list(hyperparameter_list, ax, file_reader, tag,
                                  file_template, marker_size, marker_style,
                                  label_template, use_standardized_colors=True,
                                  add_noise_indicator=True):
  """Plots a list hyperparameters.

  Args:
    hyperparameter_list: A list of tuples containing (standard deviation, alpha)
      for plotting.
    ax: The matplotlib axis to plot on.
    file_reader: A data_io function that reads data into a dict for parsing.
    tag: The tag to extract from the data.
    file_template: The file template where data is saved. Must have placeholders
      for alpha and std components, for example:
        '/path/to/data/std{std}_alpha{alpha}_file.csv
    marker_size: The size of the marker.
    marker_style: The style for the marker.
    label_template: The label to use for each line. Can have place holders for
      alpha, std, value and tag.
    use_standardized_colors: Boolean indicating if standardized color palette
      that maps std to a color defined in common_plotting should be used.
    add_noise_indicator: Boolean indicating if noise indicator lines should be
      added to the plot.

  Returns:
    The matplotlib axes object that the scatter plots were added to.
  """
  colors = common_plotting.get_colors(len(hyperparameter_list))
  for color, (std, alpha) in zip(colors, hyperparameter_list):

    data, init_datas, noise, _ = extract_centered_data(
        std, alpha, tag, file_template=file_template, file_reader=file_reader)

    if label_template is not None:
      label = label_template.format(
          std=std, alpha=alpha, value=np.mean(init_datas), tag=tag)
    else:
      label = None

    if use_standardized_colors:
      color = common_plotting.get_standardized_color(std)

    # Scatter plot.
    ax.scatter(
        data[:, 0],
        data[:, 1],
        color=color,
        s=marker_size,
        marker=marker_style,
        label=label)
    if alpha <= 0 and add_noise_indicator:
      add_noise_indicator_lines(ax, 2*noise[0], color=color)
  return ax


def _add_meta_information_to_ax(
    ax,
    xlabel_template,
    ylabel_template,
    title,
    format_dict,
    legend_style=None):
  """Adds meta-information and styling to the axis.

  Args:
    ax: The matplotlib axes to style.
    xlabel_template: A string for the x-axis with placeholders for std, alpha
      and tag.
    ylabel_template: A string for the y-axis with placeholders for std, alpha
      and tag.
    title: A string for title.
    format_dict: A dictionary of key, value pairs containing information to
      replace in placeholders. Example: {tag:.., alpha:.., std:...}
    legend_style: Styles for the legend.

  Returns:
    A matplotlib axes to continue plotting on.
  """
  if legend_style is None:
    legend_style = DEFAULT_LEGEND_STYLE
  ax.set_xlabel(xlabel_template.format(**format_dict))
  ax.set_ylabel(ylabel_template.format(**format_dict))
  ax.set_title(title)
  ax.legend(**legend_style)

  # Equal aspect the axes to improve interpretability.
  ax.set_aspect('equal')

  # To further delimit the quadrants use black lines as the x=0 and y=0 axes.
  ax.axhline(0, color='black')
  ax.axvline(0, color='black')

  return ax


def plot_2d_visualization(
    path_to_experiments,
    stds,
    alphas,
    # Data to load and interpolate.
    tag='mean_trajectory_return',
    file_reader=data_io.read_ndjson_file,
    file_name_template=DEFAULT_FILE_NAME,
    # Figure arguments.
    figsize=(10, 10),
    ax=None,
    xlabel_template='Change in {tag} in positive direction',
    ylabel_template='Change in {tag} in negative direction',
    label_template='std={std},alpha={alpha},v={value:.2f}',
    marker_style='x',
    marker_size=200,
    # Legend arguments.
    legend_style=None,
    # Noise indicators.
    use_standardized_colors=True,
    add_noise_indicator=True):
  """Will create the 2D scatter plot to visualize an objective function.

  Method to visualize data from an experiment where we have explored an
  objective function by at randomly drawn unit norm vectors.

  Args:
    path_to_experiments: String path to where the data for plotting is saved.
    stds: A list of strings with the standard deviations to plot. Note that if
      this is a list, then alphas must be a string.
    alphas: A list of strings with the alphas to plot. Note that if this is a
      list, then stds must be a string.
    tag: The data to extract from the files.
    file_reader: Reader for the file type.
    file_name_template: A string with placeholders for std and alpha.
    figsize: A tuple with the size of the figure to create if ax is not
      provided.
    ax: A matplotlib axes to plot on.
    xlabel_template: A string template for the label on the x-axis.
    ylabel_template: A string template for the label on the y-axis.
    label_template: A string template for the label of the figure.
    marker_style: The matplotlib style of the marker to use.
    marker_size: The integer size of the marker to use.
    legend_style: A dictionary containing keywords for legend styling.
    use_standardized_colors: Will use the standardized color palette that maps
      std to a color defined in common_plotting.
    add_noise_indicator: Boolean indicating if noise indicator lines should be
      added to the plot.

  Raises:
    ValueError: Only one of stds or alphas should be a list.
    ValueError: If file_name_template does not have placeholders for std and
      alpha.

  Returns:
    A matplotlib axes object where the scatter plot was drawn.
  """
  if isinstance(stds, list) == isinstance(alphas, list):
    raise ValueError('Excactly one of stds or alphas must be a list.')
  if '{std}' not in file_name_template or '{alpha}' not in file_name_template:
    raise ValueError('file_name_template must have placeholder for'
                     'std and alpha.')

  if isinstance(stds, list):
    format_dict = {'alpha': alphas, 'std': '', 'tag': tag}
    title = 'alpha={} - {}'.format(alphas, tag)
    alphas = itertools.cycle([alphas])
  elif isinstance(alphas, list):
    format_dict = {'alpha': '', 'std': stds, 'tag': tag}
    title = 'std={} - {}'.format(stds, tag)
    stds = itertools.cycle([stds])

  ax = common_plotting.get_ax(ax, figsize)
  file_template = _get_file_template(path_to_experiments, file_name_template)

  # Construct hyperparameter list:
  hyperparameter_list = zip(stds, alphas)

  # Plot said list.
  ax = _plot_for_hyperparameter_list(hyperparameter_list, ax, file_reader, tag,
                                     file_template, marker_size, marker_style,
                                     label_template, use_standardized_colors,
                                     add_noise_indicator)

  # Add meta information to the axes.
  ax = _add_meta_information_to_ax(
      ax,
      xlabel_template,
      ylabel_template,
      title,
      format_dict=format_dict,
      legend_style=legend_style)
  return ax


def add_noise_indicator_lines(ax,
                              noise_level,
                              color,
                              linewidth=DEFAULT_NOISE_INDICATOR_LINEWIDTH,
                              alpha=0.75,
                              show_hminus=True,
                              show_hplus=True,
                              show_vminus=True,
                              show_vplus=True):
  """Add indicator lines for the noise onto the matplotlib axes.

  Once you have plotted the scatter plot, you can use this function to add
  vertical and horizonal lines in a consistent method easily. These indicator
  lines are often used to show noise levels.

  Args:
    ax: A matplotlib axes object.
    noise_level: A float representing the noise level.
    color: The color of the line to plot.
    linewidth: A float representing the width of the line.
    alpha: The transparency of the points being plotted.
    show_hminus: Boolean indicating if the horizontal line with negative noise
      limit should be shown.
    show_hplus: Boolean indicating if the horizontal line with positive noise
      limit should be shown.
    show_vminus: Boolean indicating if the vertical line with negative noise
      limit should be shown.
    show_vplus: Boolean indicating if the vertical line with negative noise
      limit should be shown.

  Returns:
    A matplotlib axes object with the lines drawn as specified.
  """
  linestyle = '--'
  if show_hminus:
    ax.axhline(
        -noise_level,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha)
  if show_hplus:
    ax.axhline(
        noise_level,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha)
  if show_vminus:
    ax.axvline(
        -noise_level,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha)
  if show_vplus:
    ax.axvline(
        noise_level,
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha)
  return ax


if __name__ == '__main__':
  pass
