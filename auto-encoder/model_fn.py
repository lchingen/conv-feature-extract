import numpy as np
import tensorflow as tf

from tensorflow.layers import conv2d, conv2d_transpose
from tensorflow.layers import batch_normalization as BN
from tensorflow.layers import flatten
from tensorflow.layers import dense
from tensorflow.nn import relu, leaky_relu, sigmoid, tanh

FLAGS = tf.app.flags.FLAGS


def ACT(inputs, act_fn):
    if act_fn == 'relu':
        act = relu(inputs)
    elif act_fn == 'lrelu':
        act = leaky_relu(inputs)
    elif act_fn == 'sigmoid':
        act = sigmoid(inputs)
    elif act_fn == 'tanh':
        act = tanh(inputs)
    else:
        act = inputs
    return act


def CONV(inputs, filters, kernel_size, strides, padding, is_transpose):
    if is_transpose:
        conv = conv2d_transpose(inputs=inputs,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding)
    else:
        conv = conv2d(inputs=inputs,
                      filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding)
    return conv


def CONV_BN_ACT(inputs, filters, kernel_size, strides, padding, act_fn,
                is_training, is_transpose):
    conv = CONV(inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                is_transpose=is_transpose)

    norm = BN(inputs=conv, training=is_training)
    act = ACT(inputs=norm, act_fn=act_fn)
    return act


def Encoder(inputs, is_training):
    conv_0 = CONV_BN_ACT(inputs=inputs,
                         filters=32,
                         kernel_size=[3, 3],
                         strides=[1, 1],
                         padding='same',
                         act_fn='tanh',
                         is_training=is_training,
                         is_transpose=False)

    conv_1 = CONV_BN_ACT(inputs=conv_0,
                         filters=64,
                         kernel_size=[3, 3],
                         strides=[2, 2],
                         padding='same',
                         act_fn='tanh',
                         is_training=is_training,
                         is_transpose=False)

    return conv_0, conv_1


def Decoder(code, is_training):
    tconv_0 = CONV_BN_ACT(inputs=code,
                          filters=64,
                          kernel_size=[3, 3],
                          strides=[2, 2],
                          padding='same',
                          act_fn='tanh',
                          is_training=is_training,
                          is_transpose=True)

    tconv_1 = CONV_BN_ACT(inputs=tconv_0,
                          filters=32,
                          kernel_size=[3, 3],
                          strides=[1, 1],
                          padding='same',
                          act_fn='tanh',
                          is_training=is_training,
                          is_transpose=True)

    tconv_2 = CONV(inputs=tconv_1,
                   filters=3,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   is_transpose=True)

    act_0 = ACT(inputs=tconv_2, act_fn='tanh')
    return act_0


def AE(x, mode):
    if mode == 'TRAIN':
        is_training = True
    else:
        is_training = False

    conv_0, code = Encoder(x, is_training)
    y = Decoder(code, is_training)
    return y, conv_0


def model_fn(features, mode):
    # Instantiate model
    if type(features) is dict:
        x = features['x']
    else:
        x = features

    y, code = AE(x, FLAGS.mode)

    # Loss function
    total_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, y)))

    # Outputs
    predictions = {'x': x, 'y': y, 'code': code} # code is conv_0 output

    # Mode selection
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            # Set up logging hooks
            tf.summary.scalar('Reconstruction Loss', total_loss)
            summary_hook = tf.train.SummarySaverHook(
                                save_steps=FLAGS.save_steps,
                                output_dir=FLAGS.export_path,
                                summary_op=tf.summary.merge_all()
                            )

            # Set up optimizer
            optimizer = tf.train.AdamOptimizer(FLAGS.learn_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss=total_loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=total_loss,
                                              train_op=train_op,
                                              training_hooks=[summary_hook])
        else:
            raise NotImplementedError()


def serving_input_fn():
    # Export estimator as a tf serving API
    x = tf.placeholder(dtype=tf.float32,
                       shape=[None, FLAGS.in_h, FLAGS.in_w, FLAGS.in_c],
                       name='x')
    features = {'x': x}
    #return tf.estimator.export.TensorServingInputReceiver(features, features)
    return tf.estimator.export.ServingInputReceiver(features, features)


def export_tf_model(export_path):
    estimator = tf.estimator.Estimator(model_fn, export_path)
    estimator.export_saved_model(FLAGS.export_path + '/frozen_pb',
                                 serving_input_fn)
