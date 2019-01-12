"""Group of tools to posthoc-modify a tensorflow graph.

What this mostly does it to re-assign the weights of a keras model such that
the gradient of the loss can be taken with respect to the flattened weights.
"""
import operator
import functools

import numpy as np
import tensorflow as tf


### GRAPH-MODE TENSORFLOW UTILITY.
def posthoc_update_model_to_flat_weights(model):
  """Update a keras models weights to be slices from a flattened vector.

  Use this when you need access to weights vector in Graph Mode.
  Utility function to get the flat weights from the model and then setting them
  this connects the weights in the graph so that we can use tf.hessians.

  Args:
    model: A Keras model.

  Returns:
    flat_weights: A tf.Variable containing the flattened weights.
    model: The same Keras model but with the weights reset.
  """

  # First extract all the weights and flatten them.
  all_weights = []
  layer_weight_dims = []
  for layer in model.layers:
    if 'dense' in layer.name:
      all_weights.append(tf.reshape(layer.kernel, [-1]))
      all_weights.append(tf.reshape(layer.bias, [-1]))
      layer_weight_dims.append((layer.kernel.shape, layer.bias.shape))
    else:
      layer_weight_dims.append([])
  flat_weights = tf.concat(all_weights, -1)

  # Now reset the kernels and biases to the weights sliced from flat vector.
  start_slice = 0
  for layer, weights_def in zip(model.layers, layer_weight_dims):
    tf.logging.info('Modifying layer: %s',layer)
    if 'dense' in layer.name:
      # Get the definition of the weights.
      kernel_def, bias_def = weights_def
      end_slice = start_slice + functools.reduce(operator.mul, kernel_def, 1)
      layer.kernel = tf.reshape(
          flat_weights[start_slice:end_slice], kernel_def)
      start_slice = end_slice

      tf.logging.info('> Kernel was set')
      end_slice = start_slice + functools.reduce(operator.mul, bias_def, 1)

      layer.bias = tf.reshape(
          flat_weights[start_slice:end_slice], bias_def)
      tf.logging.info('> Bias was set')
      start_slice = end_slice

      # Hack the trainable call...
      layer._trainable_weights = [layer.kernel, layer.bias]
  return flat_weights, model

### EAGER-MODE TENSORFLOW UTILITY.
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

