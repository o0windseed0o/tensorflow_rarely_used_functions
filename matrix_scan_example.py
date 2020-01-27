import tensorflow as tf
import numpy as np

'''
a = tf.zeros([4, 3, 3])
value = tf.ones([3, 3])
value = tf.expand_dims(value, axis=0)
index = tf.constant([[1]])
d = tf.tensor_scatter_nd_update(a, index, value)
'''

def step(matrix, inputs):
    #update, idx = tf.slice(inputs, [0, 0], [3, 3]), tf.slice(inputs, [0, 1], [3, 1])
    update, idx = tf.split(inputs, [3,1], axis=1)
    idx = idx[0,:]
    idx = tf.reshape(tf.cast(idx, tf.int32), [1, 1])
    update = tf.expand_dims(update, 0)
    matrix = tf.tensor_scatter_nd_update(matrix, idx, update)
    return matrix

a = tf.zeros([4, 3, 3])
values = tf.ones([4, 3, 3])

# [4,3,1]
indices = tf.constant([[[0], [0], [0]],
                 [[1], [1], [1]],
                 [[2], [2], [2]],
                 [[3], [3], [3]]], dtype=float)

# [4,3,4]
input_array = tf.concat([values, indices], axis=2)

split0, split1 = tf.split(input_array, [3,1], axis=2)

states = tf.scan(step, input_array, initializer=a)

with tf.Session() as sess:
    # sess.run(tf.initialize_all_variables())
    #sess.run(d)
    #print(d.eval())
    print(split0.eval())
    print(split1.eval())
    #input()
    sess.run(states)
    print(input_array.eval())
    #input()
    print(states.eval())

