"""Run proposed method on a neural network."""
import time
import random
import os

from absl import flags
import numpy as np
import tensorflow as tf

import utils
import sampler_tools
import posthoc_tools

FLAGS = flags.FLAGS
gfile = tf.gfile

flags.DEFINE_string('device', 'gpu:0',
                    'Device to run graph on..')
flags.DEFINE_integer('batch_size', 1024,
                     'Batch size to use for producing the Hessian.')
flags.DEFINE_integer(
        'n_samples', 2048, 'Number of samples to take in the method.')
flags.DEFINE_float('step_size', 0.1, 'Step size for the sampling procedure.')
flags.DEFINE_integer('n_hidden', None,
        ('Hidden units. Inferred from the FLAGS.load_directory '
        'string if not provided.'))
flags.DEFINE_string('load_directory',
        None, 'Directory with saved model.')
flags.DEFINE_string('save_directory',
        os.getenv('SCRATCH', './'), 'Default save directory.')

def main(_):
  SEED = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
  tf.logging.set_verbosity(tf.logging.FATAL)  # Use FATAL here because graham.
  tf.logging.info('Running proposed method on simple neural network.')
  tf.set_random_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)
  (X_train, Y_train), _, shapes = utils.load_data()
  print(X_train.shape)
  # Use the same amount of data as in the Hessian estimation.
  sliced_dataset_size = int(0.05*len(Y_train))
  FLAGS.n_hidden = int(FLAGS.load_directory.split('hunits')[1].split('/')[0])
  tf.logging.info('n_hidden units overwritten.')

  # Create folder for saving data.
  foldername = (
          '-'.join(FLAGS.load_directory.split('/')[-3:]) + \
          '-stepsize' + str(FLAGS.step_size))
  FLAGS.save_directory = os.path.join(FLAGS.save_directory, foldername)
  if not os.path.exists(FLAGS.save_directory):
    os.makedirs(FLAGS.save_directory)
  tf.logging.info('Saving to %s', FLAGS.save_directory)

  model = utils.get_model(shapes, FLAGS.n_hidden, device=FLAGS.device)
  # We will now compile and print out a summary of our model.
  model.compile(loss='categorical_crossentropy',
              optimizer=tf.train.MomentumOptimizer(0.0, 0.0),
              metrics=['accuracy'])
  print(model.summary())
  def L_fn_mnist(weights):
    # Closure that allows us to get information about the loss function.
    posthoc_tools.set_flat_params(model, weights)

    loss, _ = model.evaluate(
        x=X_train[:sliced_dataset_size],
        y=Y_train[:sliced_dataset_size],
        verbose=0)
    return loss

  model.load_weights(FLAGS.load_directory)
  weights = posthoc_tools.get_flat_params(model.variables)
  posthoc_tools.set_flat_params(model, weights)
  print('Evaluating samples.')
  forward_samples, backward_samples = sampler_tools.get_sampled_loss_function(
    L_fn_mnist,
    weights,
    step_size=FLAGS.step_size,
    num_samples=FLAGS.n_samples,
    x0_samples=100)
  print('Sampled evaluated. Now projecting.')
  curvature_projection = sampler_tools.get_curvature_projection(
          np.array([forward_samples, backward_samples]).T)
  gradient_projection = sampler_tools.get_gradient_projection(
          np.array([forward_samples, backward_samples]).T)
  print('Projecting complete. Now saving.')
  tf.logging.info('Saving to numpy.')
  np.save(
    os.path.join(FLAGS.save_directory, 'forward_samples-{}.npy'.format(SEED)),
    forward_samples)
  np.save(
    os.path.join(FLAGS.save_directory, 'backward_samples-{}.npy'.format(SEED)),
    backward_samples)
  np.save(
    os.path.join(FLAGS.save_directory, 'curvature_projection-{}.npy'.format(SEED)),
    curvature_projection)
  np.save(
    os.path.join(FLAGS.save_directory, 'gradient_projection-{}.npy'.format(SEED)),
    gradient_projection)
  tf.logging.info('Saved.')

if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.app.run(main)

