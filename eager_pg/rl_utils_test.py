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

"""Tests to ensure rl_utils calculations are correct."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from eager_pg import rl_utils


class RLUtilsTest(absltest.TestCase):

  def test_calculate_returns(self):
    """Testing if calculations match hand-written calculations."""
    rewards = np.array([[0, 1, 0, 1],
                        [0, 1, 0, 1],
                        [0, 1, 1, 1],
                        [1, 1, 0, 1]], dtype=np.float32)  # pyformat:disable
    discount = 0.5
    masks = np.array([[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 0],
                      [1, 1, 0, 0]], dtype=np.bool)  # pyformat: disable

    calculated_returns = rl_utils.compute_discounted_return(
        rewards, discount, masks)

    self.assertEqual(calculated_returns.shape, (4, 4))

    expected_returns = np.array([[0.125, 1.875, 0.25, 1.5],
                                 [0.25, 1.75, 0.5, 1.0],
                                 [0.5, 1.5, 1.0, 0.0],
                                 [1.0, 1.0, 0.0, 0.0]])  # pyformat: disable
    self.assertTrue(np.allclose(expected_returns, calculated_returns))


if __name__ == '__main__':
  absltest.main()
