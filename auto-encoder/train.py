import numpy as np
import tensorflow as tf

from config import *
from model_fn import *
from utils import Utils

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    # Create exstimator
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=FLAGS.export_path)

    estimator.train(Utils.train_input_fn_from_tfr())


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('mode', 'TRAIN', '')
    tf.app.run()
