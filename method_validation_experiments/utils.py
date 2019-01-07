"""Utilities for simple training."""
import os

import numpy as np
import tensorflow as tf


def load_data(batch_size=64):
  """Load fashion MNIST and required variables."""
  ((X_train, Y_train),
   (X_test, Y_test)) = tf.keras.datasets.fashion_mnist.load_data()

  dataset_size = len(X_train)
  output_size = 10

  # Convert the array to float32 and normalize by 255
  # Add a dim to represent the RGB index for CNNs
  X_train = np.expand_dims((X_train.astype(np.float32) / 255.0), -1)
  X_test = np.expand_dims((X_test.astype(np.float32) / 255.0), -1)
  image_size = X_train.shape[1:]

  Y_train = (
          tf.keras.utils.to_categorical(
              Y_train, output_size).astype(np.float32))
  Y_test = (
          tf.keras.utils.to_categorical(
              Y_test, output_size).astype(np.float32))

  # FULL_DATASET_SIZE = X_train.shape[0]

  # original_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
  # dataset = original_dataset.batch(batch_size)

  # dataset = dataset.prefetch(1)
  # dataset = dataset.repeat()

  # full_dataset = original_dataset.batch(FULL_DATASET_SIZE)
  return ((X_train, Y_train),
          (X_test, Y_test),
          {'input_size': image_size,
           'output_size': output_size})

def get_model(shapes, hidden_units=0, device='gpu:0', hidden_activation=None):
  with tf.device(device):
    layers = []
    layers.append(tf.keras.layers.Flatten(input_shape=shapes['input_size']))
    if hidden_units > 0:
      layers.append(tf.keras.layers.Dense(
          hidden_units, activation=hidden_activation))
    layers.append(tf.keras.layers.Dense(
        shapes['output_size'], activation=tf.nn.softmax))

    return tf.keras.Sequential(layers)

def get_callbacks(save_directory='./', experiment_name='experiment', hidden_units=0):
  directory = os.path.join(
          save_directory,
          experiment_name,
          'checkpoints-hunits' + str(hidden_units))
  if not os.path.exists(directory):
    os.makedirs(directory)

  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(directory, '{epoch:02d}-{val_loss:.2f}.weights'),
        save_weights_only=True,
        verbose=1,
        period=5,
        )
  ]
  return callbacks


if __name__ == '__main__':
  load_data()
  get_model({'input_size':(784, 1), 'output_size': 10}, device='cpu:0')
  get_callbacks()

