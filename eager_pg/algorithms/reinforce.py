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

"""An implementation of the REINFORCE algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np  # pylint: disable=unused-import
from eager_pg import rl_utils
from eager_pg import train_utils
from eager_pg import trajectory_collector
import tensorflow as tf


class REINFORCE(train_utils.Trainer):
  """Implementation of REINFORCE (Williams, 1992).

  Each step of the algorithm proceeds as follows:
    - Collect N trajectories using a policy pi.
    - Calculate the returns from the rewards collected in those trajectories.
    - Calculate the REINFORCE loss using the trajectories.
    - Backpropagate the loss to the parameters of pi.
    - Apply gradients.
  """

  def get_gradients_and_recordables(self):
    """Collects rollouts, calculates the loss and returns gradients to apply."""
    self._start_time = time.time()
    with tf.GradientTape() as tape:
      (rewards, log_probs, masks,
       entropies) = trajectory_collector.collect_trajectories(
           self.env,
           self.policy,
           env_spec=self.env_spec,
           max_steps=self.max_steps)

      returns = rl_utils.compute_discounted_return(rewards, self.discount,
                                                   masks)
      loss = self.objective(log_probs=log_probs, returns=returns, masks=masks)
      grads = tape.gradient(loss, self.policy.trainable_variables)

      # End the timing for collecting trajectories, and the subsequent
      # backward pass.
      self._end_time = time.time()

      # Assert that grads are not all 0. This is to ensure we don't have bugs
      # in gradients.
      assert not train_utils.all_tensors_zero(grads), \
          'Some gradients were zero. {}'.format(grads)

    prepared_recordables = {
        'rewards': rewards,
        'returns': returns,
        'loss': loss,
        'masks': masks,
        'entropies': entropies
    }

    return train_utils.ReturnWrap(grads, {}, prepared_recordables)


if __name__ == '__main__':
  pass
