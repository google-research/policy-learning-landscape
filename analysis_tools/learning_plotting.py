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

"""Tools to simplify plotting learning curves from experiments.

Example of how to use:

```
experiment_path = '/PATH/TO/EXPERIMENTS/EnvWalker2d-v1/BS128/{lr}/{std}/'
hyperparameter_list = [
    # Std, alpha
    PGHyperparameter(1.0, 0.0005),
    PGHyperparameter(0.75, 0.0005),
    PGHyperparameter(0.5, 0.0001),
]

ax = learning_plotting.plot_learning_curves(hyperparameter_list,
                    experiment_path,
                    smoothing_weight=0.9,
                    x_tag='env_steps_at_deterministic_eval', xlims=(0, 5e8))
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import common_plotting
import data_io
import data_processing


DEFAULT_LEGEND_STYLE = dict(fontsize='x-small')


def _get_path(experiment_template, hyperparameter):
  """Construct a path using hyperparameters from the template."""
  return experiment_template.format(
      std=hyperparameter.std, lr=hyperparameter.lr)


# Simple abstraction to create file paths easily.
_Hyperparameter = collections.namedtuple('_Hyperparameter',
                                         'std_float lr_float')


class PGHyperparameter(_Hyperparameter):
  """Abstraction of a namedtuple to return."""

  @property
  def std(self):
    """Get a string standard deviation as it would appear in a file path."""
    return 'Std' + str(self.std_float)

  @property
  def lr(self):
    """Get a string learning rate as it would appear in a file path."""
    return 'LR' + str(self.lr_float)


def generate_hyperparameters(stds, lrs):
  """Convenience function to quickly generate hyperparameter combinations."""
  hyperparameter_values = itertools.product(stds, lrs)
  return (PGHyperparameter(*x) for x in hyperparameter_values)


def _add_meta_information_to_ax(ax,
                                title,
                                x_label_template,
                                y_label_template,
                                legend_style=None,
                                format_dict=None,
                                xlims=None,
                                ylims=None):
  """Add meta information to an axes object.

  Args:
    ax: The matplotlib axes object to add information to.
    title: The title for the plot.
    x_label_template: The template string label for the x-axis.
    y_label_template: The template string label for the y-axis.
    legend_style: A dictionary of containing kwargs for the legend.
    format_dict: A dictionary containing information to substitute into any
      templates.
    xlims: A tuple of floats or None representing how much of the x-axis to
      show.
    ylims: A tuple of floats or None representing how much of the y-axis to
      show.

  Returns:
    A Matplotlib axes object.
  """
  if format_dict is None:
    format_dict = {}
  if legend_style is None:
    legend_style = DEFAULT_LEGEND_STYLE
  ax.set_title(title)
  ax.legend(**legend_style)
  ax.set_xlabel(x_label_template.format(**format_dict))
  ax.set_ylabel(y_label_template.format(**format_dict))
  if ylims is not None:
    ax.set_ylim(*ylims)
  if xlims is not None:
    ax.set_xlim(*xlims)

  return ax


def plot_learning_curves(hyperparameter_list,
                         experiment_template,
                         y_tag='mean_deterministic_trajectory_reward',
                         x_tag='env_steps_at_deterministic_eval',
                         min_trajectory_len=0,
                         ax=None,
                         figsize=(10, 8),
                         smoothing_weight=0.9,
                         label_template='{std} - {lr}',
                         ylims=(0, 2000),
                         xlims=None,
                         x_label_template=None,
                         y_label_template='{y_tag}',
                         title_template='Hyperparameter Search Results - {env}',
                         legend_style=None,
                         shadow_standard_deviations=1,
                         shadow_alpha=0.2,
                         use_standardized_colors=True):
  """Plot learning curves for a list of hyperparameters.

  Args:
    hyperparameter_list: A list of `PGHyperparameter`'s to plot curves for.
    experiment_template: The path with where the experiments are saved. Must
      have placeholders for std and lr. Example: 'PATH/{lr}/batchsize/{std}'.
    y_tag: A string representing what to plot on the y-axis.
    x_tag: None, or a string representing what to plot on the x-axis. Valid
      string can be `env_steps_at_deterministic_eval`. If None, the x-axis will
      represent integers corresponding to the update number.
    min_trajectory_len: The minimum length of the trajectory to consider during
      averaging. Setting this to 0 will truncate all curves to the shortest one.
      If x_tag is `env_steps_at_deterministic_eval` this value represents the
      minimum number of environment steps.
    ax: A matplotlib axes to plot on.
    figsize: If no ax is provided, this is a tuple of the figure size to create.
    smoothing_weight: A float representing how much smoothing to apply.
    label_template: The template for the label. Can have placeholders for {std}
      and {lr}
    ylims: A tuple with floats representing the limits on the y-axis.
    xlims: A tuple (or None) with floats representing the limits on the x-axis.
    x_label_template: The string template for the x-axis. Usually automatically
      determined by what x_tag is.
    y_label_template: A string template of the y-axis. Defaults to `{y_tag}`.
    title_template: A string template for the title.
    legend_style: A dictionary with style to apply for the legend.
    shadow_standard_deviations: Number of standard deviations for the shadows.
    shadow_alpha: Transparency for the shadow bars.
    use_standardized_colors: Will use the standardized color palette that maps
      std to a color defined in common_plotting.

  Returns:
    A matplotlib axes object.
  """
  ax = common_plotting.get_ax(ax, figsize)
  colors = common_plotting.get_colors(len(hyperparameter_list))

  for color, hyperparameter in zip(colors, hyperparameter_list):
    experiment_path = _get_path(experiment_template, hyperparameter)
    x, y, ystderr = data_processing.extract_average_curve(
        experiment_path,
        y_tag=y_tag,
        x_tag=x_tag,
        min_trajectory_len=min_trajectory_len,
        smoothing_weight=smoothing_weight)

    label = label_template.format(std=hyperparameter.std, lr=hyperparameter.lr)
    if use_standardized_colors:
      color = common_plotting.get_standardized_color(hyperparameter.std_float)

    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x,
                    y + shadow_standard_deviations * ystderr,
                    y - shadow_standard_deviations * ystderr,
                    color=color, alpha=shadow_alpha)

  environment_name = data_io.extract_hyperparameter(experiment_template, 'Env')
  title = title_template.format(env=environment_name)

  if x_label_template is None:
    if x_tag is None:
      x_label_template = 'Number of Updates'
    elif x_tag == 'env_steps_at_deterministic_eval':
      x_label_template = 'Number of Environment Steps'
  format_dict = dict(y_tag=y_tag, x_tag=x_tag)

  ax = _add_meta_information_to_ax(
      ax,
      title,
      x_label_template,
      y_label_template,
      legend_style=legend_style,
      format_dict=format_dict,
      xlims=xlims,
      ylims=ylims)
  return ax

if __name__ == '__main__':
  pass
