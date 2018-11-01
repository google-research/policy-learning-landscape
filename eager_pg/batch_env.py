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

"""Modification of BatchEnv to support environments that loop after ending.

This BatchEnv will allow us to take steps in many environments. It also first
checks if the environment has already terminated before continuing to execute
actions. Otherwise, it returns a done and zero reward.

This is a serial version of
https://github.com/google-research/batch-ppo/blob/master/agents/tools/batch_env.py
"""  # pylint: disable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class BatchEnv(object):
  """Combine multiple environments to step in batch with support for looping

  environments.

  Looping environments are those that can come back to life (i.e. return
  done=False) after they return done=True for the first time without an explicit
  reset. This BatchEnv object will not execute actions in environments where
  actions have already been excuted.
  """

  def __init__(self, envs):
    """Combine multiple environments to step them in batch.

    Note that this does not support stepping environments in parallel.

    Args:
      envs: List of environments.

    Raises:
      ValueError: Environments have different observation or action spaces.
    """

    self._envs = envs
    observ_space = self._envs[0].observation_space
    if not all(env.observation_space == observ_space for env in self._envs):
      raise ValueError('All environments must use the same observation space.')
    action_space = self._envs[0].action_space
    if not all(env.action_space == action_space for env in self._envs):
      raise ValueError('All environments must use the same observation space.')

    self.active_envs = None  # Will keep track of with envs are active.

  def reset(self, indices=None):
    """Reset the environment and convert the resulting observation.

    Args:
      indices: The batch indices of environments to reset; defaults to all.

    Returns:
      Batch of observations.
    """

    if indices is None:
      self.active_envs = np.ones(len(self._envs))
      indices = np.arange(len(self._envs))
    else:
      self.active_envs[indices] = 1

    observs = [self._envs[index].reset() for index in indices]
    return np.stack(observs)

  def step(self, actions):
    """Forward a batch of actions to the wrapped environments.

    This will take into account that environments cannot come
    back to life after terminating. More concretely, if the environment
    returns a done, it will continue to return done until reset().

    Args:
      actions: Batched action to apply to the environment

    Raises:
      ValueError: Invalid actions
      Value Error: Callling step() before reset()

    Returns:
      Batch of observations, rewards and done flags.
    """
    if self.active_envs is None:
      raise ValueError('You must reset() before calling step().')
    for index, (env, action) in enumerate(zip(self._envs, actions)):
      if not env.action_space.contains(action):
        message = 'Invalid action at index {}: {}'
        raise ValueError(message.format(index, action))

    transitions = [
        # Execute the actions if the environment is still active.
        env.step(action) if active else None
        for env, action, active in zip(self._envs, actions, self.active_envs)
    ]

    # Create a masking_transition for inactive trajectories.
    obs_space = self._envs[0].observation_space
    if hasattr(obs_space, 'shape'):
      mask_transition = (np.zeros(obs_space.shape), 0, True, {'inactive': True})
    else:
      raise NotImplementedError('mask_transition is not implemented '
                                'for non-box observation spaces.')
    observs, rewards, dones, infos = [], [], [], []
    for env_id, env_transition in enumerate(transitions):
      if env_transition is None:
        # Environment was inactive. Put in a default transition.
        ob_, rew_, done_, info_ = mask_transition
      else:
        ob_, rew_, done_, info_ = env_transition
        if self.active_envs[env_id] and done_:
          # Environment is active and has just terminated.
          self.active_envs[env_id] = 0

      # Save the transition.
      observs.append(ob_)
      rewards.append(rew_)
      dones.append(done_)
      infos.append(info_)

    # Stack them for easy use.
    observ = np.stack(observs)
    reward = np.stack(rewards)
    done = np.stack(dones)

    info = tuple(infos)
    return observ, reward, done, info

  def close(self):
    """Send close messages to the external process and join them."""
    for env in self._envs:
      if hasattr(env, 'close'):
        env.close()

  def __len__(self):
    """Number of combined environments."""
    return len(self._envs)

  def __getitem__(self, index):
    """Access an underlying environment by index."""
    return self._envs[index]

  def __getattr__(self, name):
    """Forward unimplemented attributes to one of the original environments.

    Args:
      name: Attribute that was accessed.

    Returns:
      Value behind the attribute name one of the wrapped environments.
    """
    return getattr(self._envs[0], name)


if __name__ == '__main__':
  pass
