import os
import numpy as np
import pandas as pd
import tensorflow as tf
import src.utils as utils
from PIL import Image


class MnistDataset(object):
    def __init__(self, sess, flags, dataset_name):
        self.sess = sess
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (32, 32, 1)
        self.img_buffle = 100000  # image buffer for image shuflling
        self.num_trains, self.num_tests = 0, 0

        self.mnist_path = os.path.join('../../Data', self.dataset_name)
        self._load_mnist()

    def _load_mnist(self):
        print('Load {} dataset...'.format(self.dataset_name))
        self.train_data, self.test_data = tf.keras.datasets.mnist.load_data()
        # self.train_data is tuple: (image, label)
        self.num_trains = self.train_data[0].shape[0]
        self.num_tests = self.test_data[0].shape[0]

        # TensorFlow Dataset API
        train_x, train_y = self.train_data
        dataset = tf.data.Dataset.from_tensor_slices(({'image': train_x}, train_y))
        dataset = dataset.shuffle(self.img_buffle).repeat().batch(self.flags.batch_size)

        iterator = dataset.make_one_shot_iterator()
        self.next_batch = iterator.get_next()

        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size):
        batch_data = self.sess.run(self.next_batch)
        batch_imgs = batch_data[0]["image"]
        # batch_labels = batch_data[1]

        if self.flags.batch_size > batch_size:
            # reshape 784 vector to 28 x 28 x 1
            batch_imgs = np.reshape(batch_imgs[:batch_size], [batch_size, 28, 28])
        else:
            batch_imgs = np.reshape(batch_imgs, [self.flags.batch_size, 28, 28])

        # np.array(Image.fromarray(arr).resize())
        imgs_32 = [np.array(Image.fromarray(batch_imgs[idx]).resize(self.image_size[0:2]))
                   for idx in range(batch_imgs.shape[0])]
        # imgs_32 = [scipy.misc.imresize(batch_imgs[idx], self.image_size[0:2])
        #            for idx in range(batch_imgs.shape[0])]
        imgs_array = np.expand_dims(np.asarray(imgs_32).astype(np.float32), axis=3)

        # print('imgs shape: {}'.format(imgs_array.shape))

        return imgs_array / 127.5 - 1.  # from [0., 255.] to [-1., 1.]


class CelebA(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.image_size = (64, 64, 3)
        self.input_height = self.input_width = 108
        self.num_trains = 0

        self.celeba_path = os.path.join('../../Data', self.dataset_name, 'train')
        self._load_celeba()

    def _load_celeba(self):
        print('Load {} dataset...'.format(self.dataset_name))

        self.train_data = utils.all_files_under(self.celeba_path)
        self.num_trains = len(self.train_data)
        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size):
        batch_paths = np.random.choice(self.train_data, batch_size, replace=False)
        batch_imgs = [utils.load_data(batch_path, input_height=self.input_height, input_width=self.input_width)
                      for batch_path in batch_paths]
        return np.asarray(batch_imgs)


class TimeSeriesNaskdaq(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.file_path = "/home/wuming/repo/WGAN-TensorFlow/data/nasdaq100_padding.csv"
        # self.file_path = "/Users/wuming/CodeRepo/dl/WGAN-TensorFlow/data/nasdaq100_padding.csv"
        self.window = 32
        self.image_size = [32, 32, 1]
        self.step = 16
        self.x_trains = None

    def __preprare_ts_data__(self):
        data_frame = pd.read_csv(self.file_path)
        index_names = data_frame.columns
        sub_index_names = index_names[:self.window]
        sub_data_frame = data_frame[sub_index_names]
        # normal [-1, 1]
        normal_sub_data_frame = (sub_data_frame - sub_data_frame.min()) / (sub_data_frame.max() - sub_data_frame.min())
        normal_sub_data_frame = (normal_sub_data_frame - 0.5) * 2
        # normal_sub_data_frame = (sub_data_frame - sub_data_frame.mean()) / sub_data_frame.std()
        print(normal_sub_data_frame)
        return normal_sub_data_frame.values

    def load_data(self):
        normal_data_frame = self.__preprare_ts_data__()
        x_trains = list()
        for st in range(0, len(normal_data_frame)-self.window+1, self.step):
            ed = st + self.window
            sub_frame = normal_data_frame[st:ed]
            x_trains.append(sub_frame)
        self.x_trains = np.expand_dims(np.array(x_trains), axis=3)

    def train_next_batch(self, batch_size):
        indexes = list(range(0, len(self.x_trains)))
        select_idx = np.random.choice(indexes, batch_size, replace=False)
        frame = np.asarray(self.x_trains[select_idx])
        return frame


# noinspection PyPep8Naming
def Dataset(sess, flags, dataset_name):
    if dataset_name == 'mnist':
        return MnistDataset(sess, flags, dataset_name)
    elif dataset_name == 'celebA':
        return CelebA(flags, dataset_name)
    elif dataset_name == "nasdaq":
        return TimeSeriesNaskdaq(flags, dataset_name)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    ts_obj = TimeSeriesNaskdaq(1, "nasdaq")
    ts_obj.load_data()
