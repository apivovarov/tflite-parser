#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


#img = [1,.1,2,.2,3,.3,4,.4,5,.5,6,.6,7,.7,8,.8,9,.9,10,1,11,1.1,12,1.2]
img = [.1,.2,.3,.4,]
input_data = np.array(img).reshape(1,2,2,1)
print(input_data.shape)
print(input_data)

filter = [.1, .1,
          .1, .1,
          ]

print(np.array(filter).reshape(2,2,1,1));

# HWIM
filters = tf.constant(filter, dtype=tf.float32, shape=(2,2,1,1), name='c42')
x = tf.placeholder(dtype=tf.float32, shape=(1,2,2,1), name='input')
y = tf.nn.depthwise_conv2d(x, filters, strides=[1,1,1,1], padding='VALID')

with tf.Session() as sess:
  result = sess.run(y, feed_dict={x: input_data})

print(result.shape)
print(result)

