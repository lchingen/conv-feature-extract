import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Training configurations
tf.app.flags.DEFINE_integer('in_h',  64, '')
tf.app.flags.DEFINE_integer('in_w',  64, '')
tf.app.flags.DEFINE_integer('in_c',  3, '')

tf.app.flags.DEFINE_float('learn_rate',    0.001, '')
tf.app.flags.DEFINE_integer('num_epochs',  3, '')
tf.app.flags.DEFINE_integer('batch_size',  128, '')
tf.app.flags.DEFINE_integer('buffer_size', 50000, '')
tf.app.flags.DEFINE_integer('save_steps',  10, '')

tf.app.flags.DEFINE_string('dataset',     'celeb-face', '')
tf.app.flags.DEFINE_string('export_path', './logs', '')

tf.app.flags.DEFINE_string('train_path',  './db/train.tfrecords', '')
tf.app.flags.DEFINE_string('vld_path',    './db/train.tfrecords', '')
tf.app.flags.DEFINE_string('test_path',   './db/train.tfrecords', '')
