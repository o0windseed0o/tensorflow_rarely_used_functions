import tensorflow as tf
import numpy as np

a = tf.zeros([4, 3, 3])
value = np.ones([1, 3, 3])
index = tf.constant([[0]])
d = tf.tensor_scatter_nd_update(a, index, value)

with tf.Session() as sess:
    # sess.run(tf.initialize_all_variables())
    print(a.eval())
    sess.run(d)
    print(d.eval())