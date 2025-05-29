import tensorflow as tf
import numpy as np

a = np.arange(16)

# a = [[1,2],[3,4]]

b = tf.constant(a)

c = tf.reshape(b,(2,-1,2,2))
c = tf.transpose(c,perm=[0,2,1,3])

print(tf.shape(c))
print(c)