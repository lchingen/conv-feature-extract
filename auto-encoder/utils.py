import cv2
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS


class Utils:

    @staticmethod
    def rmse(x, y):
        assert x.shape == y.shape
        N = x.size
        print('RMSE:{}'.format(np.sqrt(np.sum((x - y)**2) / N)))

    @staticmethod
    def load_img(path, size=(64, 64), data_type=np.float32):
        img = cv2.imread(path)
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(data_type)
        return img


    @staticmethod
    def show_img(img, title=None):
        img = np.squeeze(img)
        plt.title(title)
        plt.imshow(img)
        plt.show()


    @staticmethod
    def compare(old, new):
        new, old = np.squeeze(new), np.squeeze(old)
        f = plt.figure()

        f.add_subplot(1, 2, 1, title='Old')
        plt.imshow(old)

        f.add_subplot(1, 2, 2, title='New')
        plt.imshow(new)
        plt.show()


    @staticmethod
    def compare_all(x_org, x_gen, test_size):
        f = plt.figure()
        x_grid_size = int(np.sqrt(test_size))
        y_grid_size = int(np.sqrt(test_size))
        assert x_grid_size == y_grid_size  # Just to make my life easy
        for ii in range(x_grid_size):
            for jj in range(y_grid_size):
                org = np.squeeze(x_org[x_grid_size * ii + jj])
                gen = np.squeeze(x_gen[x_grid_size * ii + jj])
                concat = np.hstack((org, gen))
                ax = f.add_subplot(y_grid_size, x_grid_size,
                                   ii * x_grid_size + jj + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(concat)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def show_all(x):
        f = plt.figure()
        x_grid_size = 8
        y_grid_size = int(np.ceil(x.shape[0] / x_grid_size))

        for ii in range(y_grid_size):
            for jj in range(x_grid_size):
                try:
                    ax = f.add_subplot(y_grid_size, x_grid_size,
                                       ii * x_grid_size + jj + 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(x[ii * x_grid_size + jj])
                except:
                    break
        plt.tight_layout()
        plt.show()


    @staticmethod
    def stream_vid(cap, size=(64, 64)):
        ret, frame = cap.read()
        frame = cv2.resize(frame, size)
        return ret, frame


    @staticmethod
    def create_dataset(path, buffer_size, batch_size, num_epochs):
        # NOTE: change the extract_feature reshape size for different datasets
        with tf.device('cpu:0'):
            dataset = tf.data.TFRecordDataset(path)\
                      .shuffle(buffer_size)\
                      .repeat(num_epochs)\
                      .map(extract_features, num_parallel_calls=4)\
                      .map(augment_features, num_parallel_calls=4)\
                      .batch(batch_size)\
                      .prefetch(1)
            return dataset


    @staticmethod
    def train_input_fn_from_tfr():
        return lambda: Utils.create_dataset(path=FLAGS.train_path,
                                      buffer_size=FLAGS.buffer_size,
                                      batch_size=FLAGS.batch_size,
                                      num_epochs=FLAGS.num_epochs)

    @staticmethod
    def pack(data, name):
        f = open(name, 'wb')
        pickle.dump(data, f)
        f.close()


    @staticmethod
    def unpack(name):
        f = open(name, 'rb')
        data = pickle.load(f)
        f.close()
        return data
