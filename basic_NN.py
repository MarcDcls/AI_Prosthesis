import tensorflow as tf
from numpy import genfromtxt

data = genfromtxt('corpus_students_only_validated_targets.csv', delimiter=',')

n_inputs =
n_outputs =

mod = tf.Sequential()
mod.add(tf.layers.Dense(256, activation='relu', input_shape=(n_inputs, )))
mod.add(tf.layers.Dense(256, activation='relu'))
mod.add(tf.layers.Dropout(0.5))
mod.add(tf.layers.Dense(64, activation='relu'))
mod.add(tf.layers.Dense(n_outputs, activation='linear'))