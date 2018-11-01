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

"""Some common multiprocessing tools for doing interpolation experiments.

These tools are specifically used to replace long running for-loops.
In particular, it executes a for-loop for a few iterations and stashes the
result into a file. In the case of `managed_multiprocessing_loop_to_numpy` data
is saved into a long numpy file. In the case of
`managed_multiprocessing_loop_to_ndjson` it is appended into a ndjson file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from multiprocessing.pool import ThreadPool
import os

import numpy as np
import landscape_explorers
import tensorflow as tf


logging = tf.logging
gfile = tf.gfile
DEFAULT_ALLOW_PICKLE = False


def managed_multiprocessing_loop_to_numpy(
    prepared_arguments,
    save_dir,
    file_name_skeleton='interpolation_result{}.npy',
    function_to_execute=landscape_explorers.landscape_explorer_parallel,
    save_every=50,
):
  """Executes a function in parallel with intermediate caching of computations.

  Given a list of tuples in `prepared_arguments`, this function executes
  `function_to_exectue` on each of them. Intermediate results are saved into
  `save_dir/file_name_skeleton` every `save_every` iterations.

  Args:
    prepared_arguments: Arguments to pass to function_to_execute. This should be
      a list of tuples.
    save_dir: The directory to save data to. Set this to be None to skip saving.
    file_name_skeleton: A file_name with a `{}` to be used for saving
      intermediate files. The final file will strip out `{}`.
    function_to_execute: The function to execute in parallel. It must take one
      argument, a tuple, that contains all the arguments it needs to run.
    save_every: The number of prepared_arguments after which to save things.

  Returns:
    Processed list of results. If save_dir is givem, it also returns the name of
    the final file saved.
  """
  pool = ThreadPool()

  logging.info('Starting multiprocessing loop.')
  processed = []
  temporary_files = []
  for i in range(0, len(prepared_arguments), save_every):
    processed += pool.map(function_to_execute,
                          prepared_arguments[i:i + save_every])

    if save_dir is not None:  # pylint: disable=pointless-statement
      temporary_files += [os.path.join(save_dir, file_name_skeleton.format(i))]

      with gfile.Open(temporary_files[-1], 'w') as f:
        np.save(f, np.array(processed))
        logging.info('Saved temporary file %s', temporary_files[-1])

      if i > 0:
        if gfile.Exists(temporary_files[-2]):
          gfile.Remove(temporary_files[-2])

  # pylint: disable=undefined-loop-variable
  # Collect the remaining arguments.
  if i + save_every <= len(prepared_arguments):
    processed += pool.map(function_to_execute,
                          prepared_arguments[i + save_every:])
  # pylint: enable=undefined-loop-variable

  logging.info('Multiprocessing loop completed. Cleaning up now.')
  pool.close()
  pool.join()

  # Delete temporary files and save data.
  if save_dir is not None:  # pylint: disable=pointless-statement
    final_file = os.path.join(save_dir, file_name_skeleton.format(''))

    with gfile.Open(final_file, 'w') as f:
      np.save(f, np.array(processed), allow_pickle=DEFAULT_ALLOW_PICKLE)
      logging.info('Saved all data.')

    if gfile.Exists(temporary_files[-1]):
      gfile.Remove(temporary_files[-1])

  logging.info('Done loop.')

  if save_dir is not None:
    return processed, final_file
  else:
    return processed


def stash_to_ndjson(results, file_name):
  """Save result into a ndjson file.

  Args:
    results: A list of dicts containing the data to save.
    file_name: The file name to save the data into.
  """
  logging.log_first_n(logging.INFO, 'Stashing to file: %s', 1, file_name)
  with gfile.Open(file_name, 'a') as writer:
    for result in results:
      writer.write(json.dumps(result))
      writer.write('\n')


def managed_multiprocessing_loop_to_ndjson(
    prepared_arguments,
    save_dir,
    file_name_skeleton='interpolation_result.ndjson',
    function_to_execute=landscape_explorers.paired_landscape_explorer_parallel,
    stash_results=stash_to_ndjson,
    save_every=10,
):
  """Executes a function parallely stashes the results into a ndjson file.

  The main advantage of this over the managed_multiprocessing_loop is that you
  can run many processes of this and save into a common file without needing to
  worry too much about preemptiveness.

  Args:
    prepared_arguments: Arguments to pass to function_to_execute. This should be
      a list of tuples.
    save_dir: The directory to save data to. Set this to be None to skip saving.
    file_name_skeleton: A file_name with a `{}` to be used for saving
      intermediate files. The final file will strip out `{}`.
    function_to_execute: The function to execute in parallel. It must take one
      argument, a tuple, that contains all the arguments it needs to run.
    stash_results: A function that takes in the intermediate results and saves
      it into the ndjson.
    save_every: The number of prepared_arguments after which to save things.

  Returns:
    Processed list of results. If save_dir is givem, it also returns the name of
    the final file saved.
  """
  pool = ThreadPool()

  logging.info('Starting multiprocessing loop.')
  processed = []
  if save_dir is not None:
    file_name = os.path.join(save_dir, file_name_skeleton)
  for i in range(0, len(prepared_arguments), save_every):
    processed_chunk = pool.map(function_to_execute,
                               prepared_arguments[i:i + save_every])

    if save_dir is not None:  # pylint: disable=pointless-statement
      stash_results(processed_chunk, file_name)

  # pylint: disable=undefined-loop-variable
  # Collect the remaining arguments.
  if i + save_every <= len(prepared_arguments):
    processed_chunk = pool.map(function_to_execute,
                               prepared_arguments[i + save_every:])
  # pylint: enable=undefined-loop-variable

  logging.info('Multiprocessing loop completed. Cleaning up now.')
  pool.close()
  pool.join()

  # Delete temporary files and save data.
  if save_dir is not None:  # pylint: disable=pointless-statement
    stash_results(processed_chunk, file_name)

  logging.info('Done loop.')

  if save_dir is not None:
    return processed, file_name
  else:
    return processed


if __name__ == '__main__':
  pass
