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

"""Unit and integration tests for trajectory_collector and batch_env.

Contains:
- Unit/Integration tests for trajectory_collector.
- Integration test for batch_env to ensure it handles sporadic environments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import gym
import numpy as np
from eager_pg import batch_env
from eager_pg import env_spec
from eager_pg import objectives
from eager_pg import policies
from eager_pg import rl_utils
from eager_pg import trajectory_collector
import tensorflow as tf


collect_trajectories = trajectory_collector.collect_trajectories

MAX_STEPS = 150  # The max number of steps you can execute in the environment.
BATCH_SIZE = 3  # The size of the batch of environments.
ACTION_SPACE = 4  # The size of the action space.
OBS_SPACE = 7  # The size of the observation space.
SPACE_MIN = -0.5  # The minimum value in the observation and action space.
SPACE_MAX = 1.5  # The maximum value in the observation and action space.


class TrajectoryCollectorTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='mountaincar_small_std',
          env_name='MountainCarContinuous-v0',
          policy_fn=policies.NormalPolicyFixedStd,
          policy_args={'std': 0.05}),
      dict(
          testcase_name='pendulum_small_std',
          env_name='Pendulum-v0',
          policy_fn=policies.NormalPolicyFixedStd,
          policy_args={'std': 0.05}),
      dict(
          testcase_name='mountaincar_large_std',
          env_name='MountainCarContinuous-v0',
          policy_fn=policies.NormalPolicyFixedStd,
          policy_args={'std': 50.0}),
      dict(
          testcase_name='pendulum_large_std',
          env_name='Pendulum-v0',
          policy_fn=policies.NormalPolicyFixedStd,
          policy_args={'std': 50.0}),
  )
  def test_collector_in_env(self, env_name, policy_fn, policy_args):
    """Will do a rollout in the environment.

    The goal of this test is two fold:
    - trajectory collections can happen.
    - action clipping happens.

    Args:
      env_name: Name of the environment to load.
      policy_fn: a policies.* object that executes actions in the environment.
      policy_args: The arguments needed to load the policy.
    """
    env = batch_env.BatchEnv([gym.make(env_name) \
                              for _ in range(BATCH_SIZE)])
    env.reset()
    policy = policy_fn(env.action_space.shape[0], **policy_args)
    spec = env_spec.EnvSpec(env)
    trajectories = trajectory_collector.collect_trajectories(
        env, policy, max_steps=MAX_STEPS, env_spec=spec)
    self.assertIsNotNone(trajectories)


class FakeEnvThatComesBackToLife(gym.Env):
  """Mock environment that comes back to life occasionally.

  The environment can randomly come back to life after returning a done.
  This basically simulates the following problem:
  t = 0: Execute action a_t --> Agent does not die. (done = False)
  t = 1: Execute action a_t --> Agent does not die. (done = False)
  t = 2: Execute action a_t --> Agent dies. (done = True)
  t = 3: Execute action a_t --> Agent comes back to life! (done = False)

  Such a trajectory can mess up calculations in our rl_utils. Since we do not
  focus on A2C style environments, we should explicitly mask out the trajectory
  for t > 3 or better yet not execute any actions.

  This environment simulates the above scenario.
  """

  def __init__(self, seed):
    """Initialize an environment."""
    self.rng = np.random.RandomState(seed)
    # pylint: disable=unused-variable
    # Disabling these because they are needed in the BatchEnv constructors.
    self.action_space = gym.spaces.Box(
        low=SPACE_MIN, high=SPACE_MAX, shape=(ACTION_SPACE,), dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=SPACE_MIN, high=SPACE_MAX, shape=(OBS_SPACE,), dtype=np.float32)
    self.at_least_one_step = False
    # pylint: enable=unused-variable

  def step(self, action):
    """Execute action in the environment.

    Args:
      action: The action to take in the environment.

    Returns:
      A tuple containing:
        - New observation (np.ndarray).
        - A reward (float).
        - Termination indicator (boolean).
        - Empty dict for compatibility.
    """
    obs = self.rng.normal(size=(OBS_SPACE,))
    # Done will be true 20% of the time.
    done = self.rng.uniform(0, 1.0) > 0.8
    if not self.at_least_one_step:
      done = False
      self.at_least_one_step = True
    return obs, 1.0, done, {}

  def reset(self):
    """Reset the environment."""
    self.at_least_one_step = False
    return np.random.normal(size=(7,))


class BatchEnvIntegrationTest(tf.test.TestCase):

  def test_batching_scheme_does_not_restart(self):
    """Test if BatchEnv correctly handle environments that come back to life."""

    env = batch_env.BatchEnv([FakeEnvThatComesBackToLife(i) \
                              for i in range(BATCH_SIZE)])
    env.reset()
    policy = policies.NormalPolicyFixedStd(ACTION_SPACE, std=0.5)
    spec = env_spec.EnvSpec(env)
    trajectories = trajectory_collector.collect_trajectories(
        env, policy, max_steps=MAX_STEPS, env_spec=spec)
    _, _, masks, _ = trajectories
    checks = []
    for t in range(1, masks.shape[0]):
      # Here the logic is:
      # (1) Find environments that were terminated (mask = 0) in the previous
      # time step.
      # (2) Check that in the current time step they are still terminated.
      # At the end we check if this was true for every time pair.
      # We expect that it will be True when trajectoies do not come back to
      # life.
      prev_time_step_end = np.where(masks[t - 1] == 0)
      checks.append(np.all(masks[t, prev_time_step_end] == 0))

    # assert that no environments came back to line.
    self.assertTrue(np.all(checks))


class TestRepeatedTrajectoryCollector(tf.test.TestCase, parameterized.TestCase):

  def test_pad_and_stack_tensors(self):
    """Test that the padding/stacking mechanism works."""
    # Width of all arrays must be the same to represent the "batch_size"
    # argument.
    my_arrays = [
        np.array([[4, 2, 1], [2, 1, 1]]),  # 2 x 3.
        np.array([[4, 2, 1], [2, 1, 1], [1, 1, 7]]),  # 3 x 3.
        np.array([[1, 2, 3]])  # 1 x 3.
    ]  # pyformat: disable

    padded_and_stacked = trajectory_collector.pad_and_stack_tensors(my_arrays)

    # Maximum trajectory length was 3 so we expect the output to be that.
    self.assertEqual(padded_and_stacked.shape, (3, 9))

    # Check that padding should only happen where it is required.
    self.assertTrue(np.allclose(padded_and_stacked[1:, 6:], 0))
    self.assertFalse(np.allclose(padded_and_stacked[:, 3:7], 0))

  @parameterized.named_parameters(
      dict(
          testcase_name='mountaincar',
          env_name='MountainCarContinuous-v0',
          policy_fn=policies.NormalPolicyFixedStd,
          policy_args={'std': 0.05}),
      dict(
          testcase_name='mujoco',
          env_name='Pendulum-v0',
          policy_fn=policies.NormalPolicyFixedStd,
          policy_args={'std': 0.05}),
  )
  def test_collector_in_env(self, env_name, policy_fn, policy_args):
    """Will do a rollout in the environment.

    The goal of this test is two fold:
    - trajectory collections can happen.
    - action clipping happens.

    Args:
      env_name: Name of the environment to load.
      policy_fn: a policies.* object that executes actions in the environment.
      policy_args: The arguments needed to load the policy.
    """
    env = batch_env.BatchEnv([gym.make(env_name) \
                              for _ in range(BATCH_SIZE)])
    env.reset()
    policy = policy_fn(env.action_space.shape[0], **policy_args)
    spec = env_spec.EnvSpec(env)
    trajectories = trajectory_collector.repeat_collect_trajectories(
        env, policy, BATCH_SIZE * 5, max_steps=MAX_STEPS, env_spec=spec)
    self.assertIsNotNone(trajectories)
    self.assertEqual(trajectories[0].shape[1], BATCH_SIZE * 5)

  def test_repeated_trajectory_collector_has_gradients(self):
    """Make sure concatenating trajectories maintains gradient information."""
    env = batch_env.BatchEnv(
        [gym.make('Pendulum-v0') for _ in range(BATCH_SIZE)])
    env.reset()
    policy = policies.NormalPolicyFixedStd(env.action_space.shape[0], std=0.5)
    spec = env_spec.EnvSpec(env)
    objective = objectives.REINFORCE()
    with tf.GradientTape() as tape:
      (rewards, log_probs, masks,
       _) = trajectory_collector.repeat_collect_trajectories(
           env,
           policy,
           n_trajectories=BATCH_SIZE * 5,
           env_spec=spec,
           max_steps=100)
      returns = rl_utils.compute_discounted_return(rewards, 0.99, masks)
      loss = objective(log_probs=log_probs, returns=returns, masks=masks)
      grads = tape.gradient(loss, policy.trainable_variables)

    self.assertTrue(len(grads))
    self.assertFalse(np.all([np.all(t.numpy() == 0) for t in grads]))


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
