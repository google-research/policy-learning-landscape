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

"""Simple tools to read data from experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow as tf

from tensorboard.backend.event_processing import event_accumulator

gfile = tf.gfile
DEFAULT_ALLOW_PICKLE = True


def get_replicates(path):
  """Get replicates of the experiment."""
  return gfile.Glob(os.path.join(path, 'Seed*'))


def extract_hyperparameter(file_path, name, delimiter='/'):
  """Extract hyperparameter value from the file path.

  Example 1:
    path: '/../learning_rate=420,momentum=101/Seed45'
    name: 'learning_rate='
    delimiter: ','
    return: '420'

  Example 2:
    path: '/../learning_rate420/momentum101/Seed45'
    name: 'learning_rate'
    delimiter: '/'
    return: '420'

  Args:
    file_path: A string file path to extract information from.
    name: The name of the hyperparameter as it appears in the path.
    delimiter: The separation between hyperparameters.

  Returns:
    The extracted hyperparameter value.
  """
  return file_path.split(name)[1].split(delimiter)[0]


## NUMPY CONVENIENCE FUNCTIONS.
def np_gload(file_path, allow_pickle=DEFAULT_ALLOW_PICKLE):
  """Load a numpy object from a file_path in a compatible format."""
  with gfile.Open(file_path, 'r') as f_:
    return np.load(f_, allow_pickle=allow_pickle)


def np_gsave(file_path, to_save, allow_pickle=DEFAULT_ALLOW_PICKLE):
  """Save a numpy object to a file_path in a compatible format."""
  with gfile.Open(file_path, 'w') as f_:
    np.save(f_, to_save, allow_pickle=allow_pickle)




def read_ndjson_file(file_path):
  """Read ndjson file from disk.

  Args:
    file_path: The path to load the data from.

  Returns:
    A list containing dictionaries of data points.
  """
  with gfile.Open(file_path, 'r') as reader:
    return [json.loads(r) for r in reader]


def extract_pairs(data_list, tag='mean_trajectory_return'):
  """Extract data from positive and negative alpha evaluations.

  Args:
    data_list: A list of dictionaries with data. Dictionary should contain 'pos'
      and 'neg' keys which represent going forward and backward in the parameter
      space. Each key should contain a dictionary of data that can be indexed by
      tag.
    tag: The key to index the data with.

  Returns:
    A list of tuples containing the extracted data.
  """
  extracted = []
  for data in data_list:
    extracted.append((data['pos'][tag], data['neg'][tag]))
    extracted.append((data['neg'][tag], data['pos'][tag]))
  return extracted


## TENSORBOARD EVENT FILE EXTRACTORS
def load_events(file_path):
  """Load the EventAccumulator object to get data from event files.

  Args:
    file_path: Path to the event files.

  Returns:
    A loaded tensorboard....event_accumulator.EventAccumulator object.
  """
  ea = event_accumulator.EventAccumulator(file_path)
  ea.Reload()
  return ea


def extract_np_from_scalar_events(scalar_events):
  """Extract steps and values from a list of scalar events.

  Args:
    scalar_events: A list of
      tensorboard....event_accumulator.ScalarEvent objects that have .value and
      .step properties that you want to extract.

  Returns:
    Two numpy arrays containing the steps and values extracted from the list.
  """
  values = []
  steps = []
  for scalar_event in scalar_events:
    values.append(scalar_event.value)
    steps.append(scalar_event.step)
  return np.array(steps), np.array(values)


if __name__ == '__main__':
  pass
