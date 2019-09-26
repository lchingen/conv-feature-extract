import numpy as np
import tensorflow as tf
from pathlib import Path


if __name__ == '__main__':
    # get latest meta and checkpoint
    subdirs = [x for x in Path('./logs').iterdir() if 'meta' in str(x)]
    latest_meta = str(sorted(subdirs)[0])

    with tf.Session() as sess:
        # restore weights
        saver = tf.train.import_meta_graph(latest_meta)
        saver.restore(sess, tf.train.latest_checkpoint('./logs/'))

        weights = tf.trainable_variables()
        weight_vals = sess.run(weights)

        # print all the weight variable names
        for name, val in zip(weights, weight_vals):
            print('-----------------------------------------------------------')
            print("name: {}, value: {}".format(name, val))

        # get first layer conv weights
        first_conv_layer_weights = weight_vals[0]
