#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


img = [1,2,3,4,5,6,7,8,9,10,11,12]
input_data = np.array(img).reshape(1,3,4,1)
print(input_data.shape)
print(input_data)

filter = [.2, .2, .1,
          .1, .2, .1,
          .3, .2, .1,
          .1, .2, .1]

print(np.array(filter).reshape(2,2,1,3));


filters = tf.constant(filter, dtype=tf.float32, shape=(2,2,1,3), name='c42')
x = tf.placeholder(dtype=tf.float32, shape=(1,3,4,1), name='input')
y = tf.nn.conv2d(x, filters, strides=[1,1,1,1], padding='VALID')

with tf.Session() as sess:
  result = sess.run(y, feed_dict={x: input_data})

print(result.shape)
print(result)

1*0.2 + 2*0.5+3*0.6+4*0.9

1*0.2 + 4*0.5+2*0.6+5*0.9
