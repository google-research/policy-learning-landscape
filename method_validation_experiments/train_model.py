"""Train a simple model on FashionMNIST."""
import time
import random
import os

from absl import flags
import numpy as np
import tensorflow as tf

import utils

FLAGS = flags.FLAGS
gfile = tf.gfile

flags.DEFINE_string('experiment_name', 'experiment',
                    'Name of the experiment.')
flags.DEFINE_string('device', 'gpu:0',
                    'Device to run graph on..')
flags.DEFINE_integer('n_hidden', 0,
                     'Number of hidden units. If zero, the model is linear.')
flags.DEFINE_integer('epochs', 100,
                     'Number of epochs to train for.')

hidden_units_map = [0, 12, 24, 36, 48, 52, 64]

def main(_):
  TASK_ID = int(os.getenv('SLURM_ARRAY_TASK_ID', -1))
  tf.logging.set_verbosity(tf.logging.INFO)
  if TASK_ID >= 0:
    FLAGS.n_hidden = hidden_units_map[TASK_ID]
    tf.logging.info('hidden units updated to: %d', FLAGS.n_hidden)
  tf.logging.info('Training simple neural network.')

  tf.set_random_seed(0)
  np.random.seed(0)
  random.seed(0)
  (X_train, Y_train), _, shapes = utils.load_data()
  model = utils.get_model(shapes, FLAGS.hidden_units, device=FLAGS.device)
  callbacks = utils.get_callbacks(FLAGS.hidden_units)

  # We will now compile and print out a summary of our model.
  model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

  model.summary()

  tf.logging.info('Training has begun.')
  model.fit(
    x=X_train,
    y=Y_train,
    batch_size=1024,
    epochs=FLAGS.epochs,
    callbacks=callbacks,
    validation_split=0.1)

  tf.logging.info('Training has ended.')

if __name__ == '__main__':
  tf.app.run(main)

