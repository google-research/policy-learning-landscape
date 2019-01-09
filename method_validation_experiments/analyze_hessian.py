"""Train a simple model on FashionMNIST."""
import time
import random
import os

from absl import flags
import numpy as np
import tensorflow as tf

import utils
import posthoc_tools

FLAGS = flags.FLAGS
gfile = tf.gfile

flags.DEFINE_string('device', 'gpu:0',
                    'Device to run graph on..')
flags.DEFINE_integer('batch_size', 1024,
                     'Batch size to use for producing the Hessian.')
flags.DEFINE_integer('n_hidden', None,
        ('Hidden units. Inferred from the FLAGS.load_directory '
        'string if not provided.'))
flags.DEFINE_string('load_directory',
        None, 'Directory with saved model.')
flags.DEFINE_string('save_directory',
                    os.getenv('SCRATCH', './'),
                    'Default save directory.')
def main(_):
  tf.logging.info('Analyzing hessian of simple neural network.')
  tf.set_random_seed(0)
  np.random.seed(0)
  random.seed(0)
  (X_train, Y_train), _, shapes = utils.load_data()

  FLAGS.n_hidden = int(FLAGS.load_directory.split('hunits')[1].split('/')[0])
  tf.logging.info('n_hidden units overwritten.')

  # Create folder for saving data.
  foldername = '-'.join(FLAGS.load_directory.split('/')[-3:])
  FLAGS.save_directory = os.path.join(FLAGS.save_directory, foldername)
  if not os.path.exists(FLAGS.save_directory):
    os.makedirs(FLAGS.save_directory)
  tf.logging.info('Saving to %s', FLAGS.save_directory)

  model = utils.get_model(shapes, FLAGS.n_hidden, device=FLAGS.device)
  (flat_weights,
   model) = posthoc_tools.posthoc_update_model_to_flat_weights(model)

  # We will now compile and print out a summary of our model.
  model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  model.load_weights(FLAGS.load_directory)

  tf.logging.info('Weights were loaded from %s', FLAGS.load_directory)
  Y_TRUE_PH = tf.placeholder(
    Y_train.dtype,
    shape=(None, *Y_train.shape[1:]))
  X_TRUE_PH = tf.placeholder(
    X_train.dtype,
    shape=(None, *X_train.shape[1:]))

  hess = tf.hessians(
    tf.reduce_mean(model.loss_functions[0](
        Y_TRUE_PH, model(X_TRUE_PH))),
    flat_weights)
  eigenvalues, eigenvectors = tf.linalg.eigh(hess)

  tf.logging.info('Model ready for Hessian/eigen computations.')
  sess = tf.keras.backend.get_session()
  tf.logging.info('Running session now...')
  hess_eval, eigenvalues_eval, eigenvectors_eval = sess.run(
          (hess, eigenvalues, eigenvectors),
          feed_dict={
              Y_TRUE_PH: Y_train[:int(0.05*len(Y_train)],
              X_TRUE_PH: X_train[:int(0.05*len(Y_train)]})
  tf.logging.info('Session ran successfully.')
  print(hess_eval)
  print(eigenvalues_eval)
  print(eigenvectors_eval)
  tf.logging.info('Saving to numpy.')
  np.save(os.path.join(FLAGS.save_directory, 'hessian.npy'), hess_eval)
  np.save(os.path.join(FLAGS.save_directory, 'eigenvalues.npy'),
          eigenvalues_eval)
  np.save(os.path.join(FLAGS.save_directory, 'eigenvectors.npy'),
          eigenvectors_eval)
  tf.logging.info('Saved.')
if __name__ == '__main__':
  tf.app.run(main)

