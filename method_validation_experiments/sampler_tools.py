"""Small library with the tools to sample the loss function."""
import numpy as np

def sample_directions(x_dim, num_samples=100):
  """Sample normalized random directions.

  Args:
    L_fn: A function that accepts a np.ndarray and returns the loss
      as a float at that point.
    x0: A np.ndarray representing the point around which to sample.
    norm: The maximum norm of the movement direction.
    num_samples: The number of samples to obtain.

  Returns:
    A np.ndarray of shape (num_samples, x_dim) such that the L2 norms are 1
    along the x_dim.
  """
  random_directions = np.random.normal(size=(num_samples, x_dim))
  random_directions /= np.linalg.norm(random_directions, axis=1).reshape(-1, 1)
  return random_directions

def get_purturbed_directions(x0, step_size=1.0, num_samples=100):
  """Get perturbed parameters.

  Args:
    x0: A np.ndarray representing the central parameter to perturb.
    step_size: A float representing the size of the step to move in.
    num_samples: The integer number of samples to draw.

  Returns:
    Two np.ndarrays representing x0 perturbed by adding a random direction and
    minusing it. They are paired so that they move by the same direction at each
    index.
  """
  directions = sample_directions(x0.shape[0], num_samples)
  forward_step_points = x0.reshape(1, -1) +  step_size * directions
  backward_step_points = x0.reshape(1, -1) -  step_size * directions
  return forward_step_points, backward_step_points

def get_sampled_loss_function(
  L_fn, x0, step_size=1.0, num_samples=100, x0_samples=1, return_points=False):
  """Sample the loss function around the perturbations.

  Args:
    L_fn: A callable function that takes a np.ndarray representing parameters
      and returns the loss.
    x0: A np.ndarray representing the central parameter to perturb.
    step_size: A float representing the size of the step to move in.
    num_samples: The integer number of samples to draw.
    x0_samples: The integer number of times to sample x0 (default is 1). Set > 1
      if the loss function is stochastic.
  """
  forward_step_points, backward_step_points = get_purturbed_directions(
    x0, step_size, num_samples)
  if x0_samples == 1:
    L_eval = L_fn(x0)
  else:
    L_eval = np.mean([L_fn(x0) for _ in range(x0_samples)])
  L_forward_eval = np.apply_along_axis(L_fn, 1, forward_step_points) - L_eval
  L_backward_eval = np.apply_along_axis(L_fn, 1, backward_step_points) - L_eval
  if return_points:
    return (
        L_forward_eval,
        L_backward_eval,
        forward_step_points,
        backward_step_points)
  else:
    return L_forward_eval, L_backward_eval

# These axes recover the gradient and curvature spectrum when projecting the 2D
# scatter plot evaluations.
CURVATURE_AX = np.array([1, 1])  # x = y
GRADIENT_AX = np.array([1, -1])  # x = -y

def scalar_project(x, v):
  """Calculate the scalar projection of vector x onto vector v."""
  v_hat = v / np.linalg.norm(v)
  return np.dot(v_hat, x)


def get_gradient_projection(values_centered):
  """Project 2D points onto the x=-y axis which gives gradient information."""
  return np.apply_along_axis(
      lambda x: scalar_project(x, GRADIENT_AX), 1, values_centered)


def get_curvature_projection(values_centered):
  """Project 2D points onto the x=y axis which gives curvature information."""
  return np.apply_along_axis(
      lambda x: scalar_project(x, CURVATURE_AX), 1, values_centered)

