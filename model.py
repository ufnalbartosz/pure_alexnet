# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from dataset import maybe_download_and_extract
dataset = maybe_download_and_extract()
train_images = dataset['train_images']
print(train_images.shape, 'images train')
train_labels = dataset['train_labels']
print(train_labels.shape)
test_images = dataset['test_images']
test_labels = dataset['test_labels']

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3], name='input')
network = conv_2d(network, 96, 11, strides=4, activation='relu', name='conv1_11_11')
network = max_pool_2d(network, 3, strides=2, name='max_pool1_3_3_2')
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu', name='conv2_5_5')
network = max_pool_2d(network, 3, strides=2, name='max_pool2_3_3_2')
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu', name='conv3_3_3')
network = conv_2d(network, 384, 3, activation='relu', name='conv4_3_3')
network = conv_2d(network, 256, 3, activation='relu', name='conv5_3_3')
network = max_pool_2d(network, 3, strides=2, name='max_pool3_3_3_2')
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh', name='fully_connected_1_tanh')
network = dropout(network, 0.5, name='dropout_1_05')
network = fully_connected(network, 4096, activation='tanh', name='fully_connected_2_tanh')
network = dropout(network, 0.5, name='dropout_2_05')
network = fully_connected(network, 17, activation='softmax', name='fully_connected_3_softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001,
                     name='target')

# Training
model = tflearn.DNN(network,
                    checkpoint_path='checkpoints/model',
                    tensorboard_verbose=3,
                    tensorboard_dir='logs')

# model.load('saves/model')
# model.load('checkpoints/model-4000')

model.fit({'input': train_images},
          {'target': train_labels},
          n_epoch=150,
          validation_set=({'input': test_images}, {'target': test_labels}),
          shuffle=True,
          show_metric=True,
          batch_size=64,
          snapshot_step=500,
          snapshot_epoch=False,
          run_id='model')

# predicted_labels = model.predict(test_images)

# pred_labels = []
# for row in predicted_labels:
#     pred_labels.append(np.argmax(row))
# true_labels = []
# for row in test_labels:
#     true_labels.append(np.argmax(row))
# num_test_images = test_images.shape[0]
# accuracy = np.sum(np.equal(pred_labels, true_labels))

# model.save('saves/model')