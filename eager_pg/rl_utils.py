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

"""Utilities for reinforcement learning.

Some tools for computing reinforcement learning specific things like discounted
returns.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def compute_discounted_return(rewards, discount, masks=None):
  """Calculates the returns from a sequence of rewards.

  Uses the recursive formulation to calculate rewards:
  G[t] = gamma*G[t+1] + reward[t]
  where G[T+1] = 0.
  Args:
    rewards: np.ndarray containing the rewards.
      Shape (trajectory_length, batch_size).
    discount: a float with the discounting per step.
    masks: np.ndarray with same shape as rewards with 0 at index i,j
      indicating that reward at i,j should be masked.
  Returns:
    The returns for each trajectory in the batch.
  """
  assert discount <= 1 and discount >= 0, 'Discount is out of allowable range.'
  assert len(rewards.shape) == 2, 'Rewards must have only two dimensions.'
  if masks is None:
    # No masks at all. Everything must be taken into account for computation.
    masks = np.ones_like(rewards, dtype=np.float32)
  trajectory_length = rewards.shape[0]
  batch_size = rewards.shape[1]
  returns = np.zeros((trajectory_length + 1, batch_size), dtype=np.float32)
  for t in reversed(range(trajectory_length)):
    returns[t] = discount * returns[t + 1] + masks[t] * rewards[t]

  return returns[:-1]


if __name__ == '__main__':
  pass
