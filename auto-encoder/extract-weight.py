import numpy as np
import tensorflow as tf
from pathlib import Path
from utils import Utils


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
    first_conv_layer_weight = weight_vals[0]
    first_conv_layer_bias = weight_vals[1]

    # pickle data
    Utils.pack(first_conv_layer_weight, './dump/weight_0.pkl')
    Utils.pack(first_conv_layer_bias, './dump/bias_0.pkl')
