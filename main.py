import tensorflow as tf
from function import *

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()


train_x = padding(train_x[:, :, :, np.newaxis, np.newaxis], 2)
print(train_x.shape)