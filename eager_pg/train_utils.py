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

"""Training utilities for policy gradient methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
from eager_pg import env_spec
from eager_pg import param_utils
from eager_pg import rl_utils
from eager_pg import trajectory_batch_stats as tb_stats
from eager_pg import trajectory_collector
import tensorflow as tf
from tensorflow.contrib import checkpoint as checkpoint_manager

gfile = tf.gfile
logging = tf.logging
DEFAULT_FREQUENCY = 1000000


CPU = env_spec.CPU
GPU = env_spec.GPU

ReturnWrap = collections.namedtuple(
    'ReturnWrap', 'grads optimizer_info recordables')


# TODO(zaf): Remove after b/111150741 is resolved.
def all_tensors_zero(tensors):
  r"""Checks if all tensors have close to zero value.

  Args:
    tensors: A list of `tf.Tensor`'s.

  Returns:
    A boolean indicating if all the tensors have close to zero value.
  """
  return np.all([np.allclose(t.numpy(), 0.0) for t in tensors])


class Trainer(object):
  """Coordinates training of a policy."""

  def __init__(self,
               env,
               policy,
               objective,
               optimizer,
               env_spec_fn=env_spec.EnvSpec,
               discount=0.995,
               max_steps=150,
               learning_rate=None,
               learning_rate_decay=None,
               deterministic_eval_frequency=DEFAULT_FREQUENCY):
    """Handles training policies.

    Will collect trajectories, calculate objectives and update policies. Will
    also take care of checkpointing and summary saving.

    Args:
      env: The batched environment to sample trajectories from. This should
        return observations as a np.ndarray of np.float32 with shape
        (batch_size, obs_size).
      policy: A policies.Policy object with learnable parameters.
      objective: A objectives.Objective object that retuns a tf.Variable to be
        _minimized_.
      optimizer: A tf.train.Optimizer instance that will be used to apply
        gradient updates to the parameters.
      env_spec_fn: A callable function that returns an env_spec.EnvSpec object
        to facillitate interfacing between gym and tensorflow.
      discount: A discount factor for the trajectory horizon (defaults to 0.99).
      max_steps: An integer for the maximum number of steps to execute the
        policy in the environment before terminating.
      learning_rate: a tf.Variable instance with the learning rate. This only
        needs to be supplied if you want to record it in the summaries or
        you want to decay it during training.
      learning_rate_decay: a function that returns the new learning_rate that
        will be assigned to the tf.Variable above.
      deterministic_eval_frequency: The frequency at which to do a deterministic
        evaluation of the policy.
    """
    self.env = env
    self.env_spec = env_spec_fn(env)
    self.max_steps = max_steps
    self.policy = policy
    self.objective = objective
    self.optimizer = optimizer
    self.discount = discount
    self.summary_writer = None
    self.checkpointer = None

    # Stores number of environment steps taken.
    self.n_environment_steps = tf.Variable(0.0, trainable=False)

    if learning_rate_decay is not None:
      assert learning_rate is not None,\
          'You must pass learning_rate if you want to decay the it.'
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay

    self.parameter_frequency = DEFAULT_FREQUENCY
    self.summary_frequency = DEFAULT_FREQUENCY
    self.checkpoint_frequency = DEFAULT_FREQUENCY
    self.deterministic_eval_frequency = deterministic_eval_frequency

  def create_checkpointer(self,
                          checkpoint_directory,
                          variables_to_checkpoint=None,
                          checkpoint_frequency=5):
    """Creates a tensorflow checkpointer, allows restart from checkpoints.

    Args:
      checkpoint_directory: String filepath to where the checkpoints will be
        saved.
      variables_to_checkpoint: A dictionary containing other variables to
        checkpoint.
      checkpoint_frequency: An integer number of updates after which to save
        checkpoints.
    """
    if variables_to_checkpoint is None:
      variables_to_checkpoint = dict()
    self.checkpoint_directory = os.path.join(
        checkpoint_directory, 'checkpoints')
    self.checkpointer = tf.train.Checkpoint(
        optimizer=self.optimizer,
        policy=self.policy,
        learning_rate=self.learning_rate,
        optimizer_step=tf.train.get_or_create_global_step(),
        n_environment_steps=self.n_environment_steps,
        **variables_to_checkpoint)
    self._checkpoint_manager = checkpoint_manager.CheckpointManager(
        self.checkpointer, directory=self.checkpoint_directory, max_to_keep=5)

    self.checkpoint_frequency = checkpoint_frequency
    logging.info('Checkpoints will be saved to %s every %d updates.',
                 self.checkpoint_directory, self.checkpoint_frequency)

  def create_parameter_saver(self, save_directory, parameter_frequency=5):
    """Set where and how frequently to save parameters.

    Args:
      save_directory: String filepath to where parameters will be saved as
        numpy files.
      parameter_frequency: An integer number of updates after which to save
        parameters.
    """
    self.parameters_directory = os.path.join(save_directory, 'parameters')
    self.parameter_frequency = parameter_frequency
    gfile.MakeDirs(self.parameters_directory)
    logging.info('parameters will be saved to %s every %d updates',
                 self.parameters_directory, self.parameter_frequency)

  def restore_checkpoint(self, checkpoint_name='latest'):
    """Will restore to the checkpoint given. By default the latest."""
    if self.checkpointer is None:
      raise AssertionError('Make sure you call create_checkpointer()'
                           ' before restoring.')
    if checkpoint_name == 'latest':
      checkpoint_name = tf.train.latest_checkpoint(self.checkpoint_directory)
    if checkpoint_name is not None:
      logging.info('Restoring from {}'.format(checkpoint_name))
      self.checkpointer.restore(checkpoint_name)

  def create_summary_writer(self, log_directory, summary_frequency=5):
    """Create summary writers for Tensorboard.

    Args:
      log_directory: String filepath to where the tensorboard events will be
        saved.
      summary_frequency: An integer number of updates after which to save
        tensorboard summaries.
    """
    self.summary_writer = tf.contrib.summary.create_summary_file_writer(
        log_directory)  # pylint: disable=line-too-long
    self.log_directory = log_directory
    self.summary_frequency = summary_frequency
    logging.info('summaries will be saved to %s every %d updates',
                 self.log_directory, self.summary_frequency)

  def maybe_checkpoint(self):
    """Will check point the training if enough updates have passed."""
    if self.checkpointer is None:
      return
    if self.should_save('checkpoints'):
      self._checkpoint_manager.save()

  def should_save(self, quantity=None):
    """Returns if something needs to be saved at this global step."""

    # Always save the first of everything.
    step = tf.train.get_or_create_global_step().numpy()
    if step == 0: return True

    if quantity == 'summaries':
      divisor = self.summary_frequency
    elif quantity == 'parameters':
      divisor = self.parameter_frequency
    elif quantity == 'checkpoints':
      divisor = self.checkpoint_frequency
    elif quantity == 'determinstic_eval':
      divisor = self.deterministic_eval_frequency
    else:
      return False

    return step % divisor == 0

  def maybe_write_summaries(self, recordables):
    """Will record tensorboard summaries from given quantities.

    Args:
      recordables: A dict containing tf.Tensor or np.ndaray's to save summaries
        of. _At_least_ the following attributes must be present:
        - rewards: A np.ndarray of shape (trajectory_length, batch_size) with
          rewards as np.float32 for each timestep and trajectory.
        - returns: A np.ndarray of shape (trajectory_length, batch_size) with
          returns as np.float32 for each timestep and trajectory.
        - loss: A tf.Variable with the loss.
        - masks: A np.ndarray of shape (trajectory_length, batch_size) with
          masking information for each timestep and trajectory.
        - entropies: A tf.Tensor of shape (trajectory_length, batch_size) with
          entropy for each timestep and trajectory.
    """
    # Obtain the minimal summarizable information.
    rewards = recordables['rewards']
    returns = recordables['returns']
    loss = recordables['loss']
    masks = recordables['masks']
    entropies = recordables['entropies']

    if self.summary_writer is None:
      return
    if not self.should_save('summaries'):
      # Do a check here to prevent calculation of all the other summaries
      # if we are not going to record them.
      return
    with self.summary_writer.as_default(), \
        tf.contrib.summary.always_record_summaries():
      # Holder for statistics to be saved.
      stats = {}

      # Information about learning rate.
      if self.learning_rate is not None:
        stats['learning_rate'] = self.learning_rate

      # Information about how long the update took.
      stats['step_time'] = self._end_time - self._start_time

      # Information about trajectory lengths.
      trajectory_lengths = tb_stats.get_trajectory_lengths(masks)
      stats['mean_trajectory_lengths'] = tf.reduce_mean(trajectory_lengths)

      # Running count of the number of environment steps taken so far.
      stats['total_environment_steps'] = self.n_environment_steps

      # Information about trajectory rewards and returns.
      stats.update(
          tb_stats.reward_summaries(
              rewards, masks, trajectory_lengths=trajectory_lengths))
      stats.update(tb_stats.return_summaries(returns))

      # Information about trajectory entropy.
      stats.update(
          tb_stats.entropy_summaries(
              entropies, masks, trajectory_lengths=trajectory_lengths))

      # Do a deterministic evaluation if necessary.
      stats.update(self.maybe_deterministic_evaluate())

      for stat_name, stat_value in stats.items():
        tf.contrib.summary.scalar(stat_name, tensor=stat_value)

      step = tf.train.get_or_create_global_step().numpy()

      logging.info(
          'Updates {}, '
          'Loss {:.5f}, '
          'Total Reward {:.2f} +- {:.2f}, '
          'Reward Per Step {:.2f}, '
          'Return {:.2f} +- {:.2f} '
          'Total Entropy {:.2f}, '
          'Entropy '
          'Per Step {:.2f}'.format(
              step, loss.numpy(), stats['mean_trajectory_reward'],
              stats['stderr_trajectory_reward'], stats['mean_step_reward'],
              stats['mean_trajectory_return'],
              stats['stderr_trajectory_return'],
              stats['mean_trajectory_entropy'], stats['mean_step_entropy']))

  def maybe_save_parameters(self, force=False):
    """Saves parameter vectors if enough updates have passed.

    Args:
      force: Force saving of parameters (boolean).
    """
    if self.should_save('parameters') or force:
      step = tf.train.get_or_create_global_step().numpy()
      parameter_vector = param_utils.get_flat_params(self.policy.variables)
      file_name = os.path.join(self.parameters_directory, '{}.npy'.format(step))
      with gfile.Open(file_name, 'w+') as file_:
        np.save(file_, parameter_vector, allow_pickle=False)

  def evaluate_policy(self, policy=None):
    r"""Evaluate a policy by doing rollouts.

    Args:
      policy: A policies.BasePolicy object to be rolled out. If None, it will
        rollout the policy being trained. If you want to rollout the
        deterministic policy use `evaluate_deterministic_policy()`.

    Returns:
      A dict containing tf.Tensor's with statistics about the rollouts. It will
      contain the keys:
        - `mean_trajectory_return`
        - `stderr_trajectory_return`
        - `mean_trajectory_reward`
        - `stderr_trajectory_reward`
    """
    if policy is None:
      policy = self.policy
    (rewards, _, masks, _) = trajectory_collector.collect_trajectories(
        self.env, policy, env_spec=self.env_spec, max_steps=self.max_steps)
    stats = tb_stats.reward_summaries(rewards, masks)
    returns = rl_utils.compute_discounted_return(rewards, self.discount, masks)
    stats.update(tb_stats.return_summaries(returns))
    return stats

  def maybe_deterministic_evaluate(self, force=False):
    r"""Evaluates the deterministic version of the policy.

    Args:
      force: A boolean indicating if the evaluation should run regardless of if
        the current iteration allows for a deterministic evaluation.

    Returns:
      A dict containing tf.Tensor's with statistics about the rollouts. It will
      contain the keys:
        - `mean_deterministic_return`
        - `stderr_deterministic_return`
        - `mean_deterministic_trajectory_reward`
        - `stderr_deterministic_trajectory_reward`

    """
    if not self.should_save('determinstic_eval') and not force:
      return {}

    policy_copy = self.policy.get_deterministic_copy()
    stats = self.evaluate_policy(policy_copy)

    # Rename keys here so that they mention 'deterministic'.
    return dict(
        env_steps_at_deterministic_eval=self.n_environment_steps,
        mean_deterministic_return=stats['mean_trajectory_return'],
        stderr_deterministic_return=stats['stderr_trajectory_return'],
        mean_deterministic_trajectory_reward=stats['mean_trajectory_reward'],
        stderr_deterministic_trajectory_reward=stats['stderr_trajectory_reward']
        )  # pyformat: disable

  def evaluate_deterministic_policy(self):
    """Evaluates the current policy in a deterministic fashion."""
    return self.maybe_deterministic_evaluate(force=True)

  def get_gradients_and_recordables(self):
    """Does rollouts and returns the gradients to apply.

    Returns:
      grads: A list of tf.Tensor with the gradients to apply to your parameters.
      optimizer_info: A dict containing any information to pass to the
        optimizer.
      recordables: A dict containing `tf.Tensor`'s and `np.ndarray`s to save
        summaries. This dict must contain _at_least_ the following entries:
          - rewards, returns, loss, masks, entropies
    """
    raise NotImplementedError('You must implement this method.')

  def apply_gradients(self, grads, optimizer_info=None):  # pylint: disable=unused-argument
    """Applies the gradients to update parameters.

    Note: In most algorithms the `optimizer_info` argument is ignored.
    It is retained here to ensure that new algorithms can be implemented easily
    without needing to modify `train()`.

    Args:
      grads: A list of tf.Tensor with the gradients to apply to your parameters.
      optimizer_info: A dict containing other data that might be relevant to the
        optimizer before applying the gradients.
    """
    self.optimizer.apply_gradients(
        zip(grads, self.policy.variables),
        global_step=tf.train.get_or_create_global_step())

  def train(self, updates):
    """Trains a policy for a given number of updates.

    Args:
      updates: The number of updates to train for (int).
    """
    for _ in range(updates):
      grads, optimizer_info, recordables = self.get_gradients_and_recordables()

      # Write summaries before doing any kind of update.
      # We do this here because we want the parameters and the quantities in
      # Tensorboard to correspond exactly.
      # Do this outside tape so we don't record things we don't need onto it.
      self.maybe_checkpoint()
      self.maybe_write_summaries(recordables)
      self.maybe_save_parameters()

      # Do the update step.
      self.apply_gradients(grads, optimizer_info)

      # Increment the amount of steps in all trajectories to get an idea of
      # data efficiency.
      trajectory_lengths = tb_stats.get_trajectory_lengths(recordables['masks'])
      self.n_environment_steps.assign_add(tf.reduce_sum(trajectory_lengths))

      if self.learning_rate_decay is not None:
        self.learning_rate.assign(self.learning_rate_decay())

      # Stopping conditions.
      # (1) Maximum number of updates reached.
      if tf.train.get_or_create_global_step() > updates:
        return

if __name__ == '__main__':
  pass
