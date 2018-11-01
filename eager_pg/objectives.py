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

"""Contains objective functions to be used in policy gradient algorithms.

Currently implemented objectives:
  - REINFORCE: from Williams, Ronald J. "Simple statistical gradient-following
    algorithms for connectionist reinforcement learning." 1999. The gradient
    this will be equal to the gradient of the ExpectedReturn.

Note that the objectives here are designed to work with batches of trajectories
and not just batches whereas policies are designed to work with just a batch
dimension.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class REINFORCE(object):
  r"""The REINFORCE "loss" function.

  Differentiating this will give the gradient of the expected discounted return.

  Calculation is as follows:
  ```
  E[R] = \sum_\tau P(\tau) G(\tau)
  ```
  where \tau is a trajectory, P is the probability of that trajectory and G
  is the return from the first state of that trajectory.
  We can then take the derivative of this with respect to the parameters of the
  policy:

  ```
  \nabla E[R] = \nabla \sum_\tau P(\tau) G(\tau)
            = \sum_\tau \nabla P(\tau) G(\tau)
  ```
  Now introduce $P(\tau)/P(\tau)$:
  ```
  = \sum_\tau P(\tau)/P(\tau) \nabla P(\tau) G(\tau)
  = \sum_\tau P(\tau) \nabla \log P(\tau) G(\tau)
  ```
  where in the last line we use the log ratio trick. This can now be sampled and
  therefore is suitable to use in a stochastic gradient ascent algorithm:
  ```
  = \sum_\tau P(\tau) \nabla \log P(\tau) G(\tau)
  \approx \frac{1}{N}\sum_n \nabla \log P(\tau) G(\tau)
  ```

  For a more thorough derivation see (Williams, 1999) or (Sutton, 2000).
  """

  def __call__(self, log_probs, returns=None, masks=None, advantages=None):
    """Calculates the differentiable loss when called as REINFORCE()(data).

    Args:
      log_probs: The tf.Tensor for the log probability of the action taken.
        Shaped (None, batch_size).
      returns:
      masks: The np.ndarray containing masks to apply to the advantages.
        Shaped (None, batch_size).
      advantages: The pre-computed advantage as an np.ndarray. Passing this in
        will not use the `returns` input. Shaped (None, batch_size).

    Raises:
      ValueError: If log_probs is not a tf.Tensor. This can happen if you pass
      in a raw list of log_probs. You must stack them using `tf.stack()`
      ValueError: If both returns and advantages are none.

    Returns:
      The computed loss as a tf.Tensor.
    """
    if not isinstance(log_probs, tf.Tensor):
      raise ValueError('log_probs must be a tf.Tensor. Make sure you use '
                       'tf.stack() to convert a list to Tensor.')
    if returns is None and advantages is None:
      raise ValueError('Both returns and advantages cannot be None.')
    if masks is None:
      masks = np.ones_like(returns, dtype=np.float32)
    if advantages is None:
      advantages_npy = masks * (returns - returns.mean()) / (
          1e-8 + returns.std())
      advantages = tf.constant(advantages_npy)
    else:
      advantages = tf.constant(advantages * masks)
    return tf.reduce_mean(-tf.reduce_sum(log_probs * advantages, 0))


if __name__ == '__main__':
  pass
