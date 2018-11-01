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

"""EnvSpec that featurizes observations using random radial basis functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from eager_pg import env_spec
from sklearn import kernel_approximation
from sklearn import pipeline

from sklearn import preprocessing as skl_preprocessing


class RBFEnvSpec(env_spec.EnvSpec):
  """EnvSpec that converts observations into RBF feature representations."""

  def __init__(self,
               env,
               sample_mode='rollouts',
               num_components=50,
               gammas=None,
               num_obs=10000,
               use_standard_scaler=True,
               featurizer_max_env_steps=1000):
    """RBF Env Spec Featurizer.

    Feature Representation given by RBF Kernels.
    See Rahimi and Recht "Random features for large-scale kernel machines." 2008

    Args:
      env: The environment to move tensors between.
      sample_mode: A string rerpresenting how to collect data from the
        environment to build features. Must be {'rollouts', 'reset', 'random'}.
        - `rollouts` will collect observations by executing a random policy in
          the env.
        - `reset` will collect rollouts by repeatedly resetting the env.
        - `random` will just sample the env observation space randomly.
      num_components: The number of components in each RBF.
      gammas: A list containing the frequency of each RBF. If None will default
        to `[0.5, 1.0, 2.5, 5.0]`.
      num_obs: The integer number of observations to use to fit the Kernels.
      use_standard_scaler: Boolean indicating if the observations should be
        normalized.
      featurizer_max_env_steps: Maximum number of steps to be taken in each
        rollout to estimate the kernels in the featurizer.

    Raises:
      ValueError: If the `sample_mode` is unknown.
    """
    super(RBFEnvSpec, self).__init__(env)
    self._build_feature_pipeline(
        sample_mode,
        num_components,
        gammas,
        num_obs,
        use_standard_scaler,
        featurizer_max_env_steps)

  def _build_feature_pipeline(self,
                              sample_mode='rollouts',
                              num_components=50,
                              gammas=None,
                              num_obs=10000,
                              use_standard_scaler=True,
                              featurizer_max_env_steps=10000):
    """Build the feature pipeline.

    Args:
      sample_mode: A string rerpresenting how to collect data from the
        environment to build features. Must be {'rollouts', 'reset', 'random'}.
        - `rollouts` will collect observations by executing a random policy in
          the env.
        - `reset` will collect rollouts by repeatedly resetting the env.
        - `random` will just sample the env observation space randomly.
      num_components: The number of components in each RBF.
      gammas: A list containing the frequency of each RBF. If None will default
        to `[0.5, 1.0, 2.5, 5.0]`.
      num_obs: The integer number of observations to use to fit the Kernels.
      use_standard_scaler: Boolean indicating if the observations should be
        normalized.
      featurizer_max_env_steps: Maximum number of steps to be taken in each
        rollout to estimate the kernels in the featurizer.

    Raises:
      ValueError: If the `sample_mode` is unknown.
    """
    env = self._env._envs[0]  # pylint: disable=protected-access
    if gammas is None:
      gammas = [0.5, 1.0, 2.5, 5.0]

    features = []
    for i, gamma in enumerate(gammas):
      features.append(
          ('rbf{}'.format(i),
           kernel_approximation.RBFSampler(
               gamma=gamma, n_components=num_components)
          ))
    self.featurizer = pipeline.FeatureUnion(features)
    if use_standard_scaler: self.scaler = skl_preprocessing.StandardScaler()

    if sample_mode == 'random':
      # Randomly sample from the observation space to fit the featurizers.
      observation_examples = np.array([env.observation_space.sample() for _ in range(num_obs)])  # pylint: disable=line-too-long
    elif sample_mode == 'reset':
      # Just reset the environment to obtain the observations.
      observation_examples = np.array([env.reset() for _ in range(num_obs)])
    elif sample_mode == 'rollouts':
      # Rollout mode.
      observations = []
      while True:
        observations.append(env.reset())
        done = False
        t = 0
        while not done and t < featurizer_max_env_steps:
          action = env.action_space.sample()
          obs, _, done, _ = env.step(action)
          observations.append(obs)
        if len(observations) > num_obs: break  # Collected enough observations.
      observation_examples = np.array(observations)
    else:
      raise ValueError('Unknown `sample_mode`!')

    if use_standard_scaler: self.scaler.fit(observation_examples)
    if use_standard_scaler: self.scaler.transform(observation_examples)
    self.featurizer.fit(observation_examples)
    self.use_standard_scaler = use_standard_scaler

  def convert_obs_gym_to_tf(self, obs, featurize=True):
    """Converts a gym observation into a tensorflow suitable format.

    Args:
      obs: A numpy array shaped (batch_size, obs_dim) representing an
        observation from a gym environment.
      featurize: A boolean indicating if the observations should be featurized.

    Returns:
      A tf.Tensor suitable to pass into a network.
    """
    if featurize:
      if self.use_standard_scaler: obs = self.scaler.transform(obs)
      obs = self.featurizer.transform(obs)
    return super(RBFEnvSpec, self).convert_obs_gym_to_tf(obs)

if __name__ == '__main__':
  pass
