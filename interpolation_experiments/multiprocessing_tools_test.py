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

"""Unit tests for multiprocessing_tools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile

import multiprocessing_tools
import tensorflow as tf


gfile = tf.gfile


# Dummy functions to execute in parallel.
def adder_function(args):
  """Dummy function to execute returning a single float."""
  loc, scale = args
  return loc + scale


def adder_function_json(args):
  """Dummy function to execute returning a dict."""
  loc, scale = args
  return {'result': loc + scale}


DEFAULT_ARGUMENTS = list(zip(range(10), range(10)))


class MultiprocessingToolsTest(tf.test.TestCase):

  def test_managed_multiprocessing_loop_to_numpy(self):
    """Test if multiprocessing loops work."""

    save_dir = tempfile.mkdtemp()

    # pylint: disable=line-too-long
    results, file_name = multiprocessing_tools.managed_multiprocessing_loop_to_numpy(
        DEFAULT_ARGUMENTS,
        save_dir=save_dir,
        function_to_execute=adder_function,
        # Reduce the save_every to ensure we get to the end of the list.
        save_every=3,
    )
    # pylint: enable=line-too-long
    self.assertEqual(file_name,
                     os.path.join(save_dir, 'interpolation_result.npy'))

    self.assertTrue(all([i * 2 == res for i, res in zip(range(10), results)]))

  def test_managed_multiprocessing_loop_to_ndjson(self):
    """Test if multiprocessing loops work."""

    save_dir = tempfile.mkdtemp()

    # pylint: disable=line-too-long
    results, file_name = multiprocessing_tools.managed_multiprocessing_loop_to_ndjson(
        DEFAULT_ARGUMENTS,
        save_dir=save_dir,
        function_to_execute=adder_function_json,
        # Reduce the save_every to ensure we get to the end of the list.
        save_every=3,
    )

    # pylint: enable=line-too-long

    # Check if the results are correct.
    self.assertTrue(all([i * 2 == res for i, res in zip(range(10), results)]))

    # Check if file was saved in the right place.
    self.assertEqual(file_name,
                     os.path.join(save_dir, 'interpolation_result.ndjson'))

    # Check if IO happened correctly.
    with gfile.Open(file_name, 'r') as reader:
      results_from_ndjson = [json.loads(r)['result'] for r in reader]

    # Check if the result in the file is correct.
    self.assertTrue(
        all([i * 2 == res for i, res in zip(range(10), results_from_ndjson)]))


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
