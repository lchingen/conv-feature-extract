import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path

from config import *
from model_fn import *
from utils import Utils

tf.enable_eager_execution()

img_path = './imgs/dog.jpg'

def main(unused_argv):
    # Export model_fn to only use decoder
    export_tf_model(FLAGS.export_path)

    # Find latest frozen pb
    subdirs = [x for x in Path(FLAGS.export_path + '/frozen_pb').iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    # Create predictor
    predict_fn = predictor.from_saved_model(latest)

    # Read image
    x  = Utils.load_img(img_path)[None] / 255.0
    dict_in = {'x': x}

    # Make predictions and fetch results from output dict
    y = predict_fn(dict_in)['y']
    code = np.squeeze(predict_fn(dict_in)['code'])
    code = np.transpose(code, (2,0,1))

    # Show all source v.s. generated results
    #Utils.rmse(x,y)
    #Utils.compare(x, y)
    Utils.show_all(code)

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('mode', 'TEST', 'TRAIN/TEST')
    tf.app.run()
