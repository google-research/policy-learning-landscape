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

"""Tools for extracting and setting parameters of a neural network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def get_flat_params(parameters):
  """Returns flattened model parameters.

  Given a list of tensorflow variables, this returns a numpy array
  containing a flat representation of all the parameters.
  Only works in eager mode.
  Args:
    parameters: The iterable containing the tf.Variable objects.
  Returns:
    A numpy array containing the parameters.
  """
  params = []
  for param in parameters:
    params.append(param.numpy().reshape(-1))
  return np.concatenate(params)


def set_flat_params(model, flat_params, trainable_only=True):
  """Set model parameters with a linear numpy array.

  Takes a flat tensor containing parameters and sets the model with
  those parameters.
  Args:
    model: The tf.keras.Model object to set the params of.
    flat_params: The flattened contiguous 1D numpy array containing
      the parameters to set.
    trainable_only: Set only the trainable parameters.
  Returns:
    The keras model from `model` but with the parameters set to `flat_params`.
  """
  idx = 0
  if trainable_only:
    variables = model.trainable_variables
  else:
    variables = model.variables

  for param in variables:
    # This will be 1 if param.shape is empty, corresponding to a single value.
    flat_size = int(np.prod(list(param.shape)))
    flat_param_to_assign = flat_params[idx:idx + flat_size]
    # Explicit check here because of: b/112443506
    if len(param.shape):  # pylint: disable=g-explicit-length-test
      flat_param_to_assign = flat_param_to_assign.reshape(*param.shape)
    else:
      flat_param_to_assign = flat_param_to_assign[0]
    param.assign(flat_param_to_assign)
    idx += flat_size
  return model


if __name__ == '__main__':
  pass
