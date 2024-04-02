# example of defining a u-net encoder-decoder generator model
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

A = np.random.rand(5,5)
B = np.random.rand(5,5)

x = K.variable(value=A)
y = K.variable(value=B)

z = K.dot(x,y)

flops = tf.profiler.profile(K.get_session().graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
params = tf.profiler.profile(K.get_session().graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
print("flops: ", flops.total_float_ops)
print("GFLOPs: ", flops.total_float_ops / 1000000000.0)
print("trainable params: ", params.total_parameters)