import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path

from config import *
from model_fn import *
from utils import Utils

tf.enable_eager_execution()


def main(unused_argv):
    # set test image path
    img_path = './imgs/dog.jpg'

    # export model_fn to only use decoder
    export_tf_model(FLAGS.export_path)

    # fetch latest frozen pb
    subdirs = [x for x in Path(FLAGS.export_path + '/frozen_pb').iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    # initiate predictor
    predict_fn = predictor.from_saved_model(latest)

    # read and normalize image
    x  = Utils.load_img(img_path)[None] / 255.0
    dict_in = {'x': x}

    # inference; ordered in (H,W,C)
    code = np.squeeze(predict_fn(dict_in)['code'])

    # dump input and IA after autoencoder
    ia = x
    oa = code

    # pickle data
    Utils.pack(ia, './dump/ia.pkl')
    Utils.pack(oa, './dump/oa_0.pkl')


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('mode', 'TEST', 'TRAIN/TEST')
    tf.app.run()
