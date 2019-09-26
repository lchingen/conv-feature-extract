import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path

from config import *
from model_fn import *
from utils import Utils


tf.enable_eager_execution()

def main(unused_argv):
    # Export model_fn to only use decoder
    export_tf_model(FLAGS.export_path)

    # Find latest frozen pb
    subdirs = [x for x in Path(FLAGS.export_path + '/frozen_pb').iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    # Create predictor
    predict_fn = predictor.from_saved_model(latest)

    # Stream image
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cv2.namedWindow('stream')

    while cap.isOpened():
        # Stream
        ret, frame = Utils.stream_vid(cap, size=(64,64))
        if not ret:
            break

        # Inference
        x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = x[None, ...] / 255.0

        prediction = predict_fn({'x':x})
        y = prediction['y'][0]
        # Just easier for me to see
        y = cv2.resize(y, (FLAGS.size, FLAGS.size))
        y = cv2.cvtColor(y, cv2.COLOR_RGB2GRAY)

        x = cv2.resize(frame, (FLAGS.size, FLAGS.size))
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = x / 255.0
        compare = np.hstack((x,y))
        #----------------------------------------------
        code = np.squeeze(prediction['code'])
        # Just easier for me to see
        code = cv2.resize(code, (FLAGS.size, FLAGS.size))
        code = np.transpose(code, (2,0,1))
        compare = np.hstack((compare, code[0]))
        # ---------------------------------------------
        cv2.imshow('stream', compare)

        # Key break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('mode', 'TEST', 'TRAIN/TEST')
    tf.app.flags.DEFINE_integer('size', 256, '')
    tf.app.run()
