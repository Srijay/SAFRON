import tensorflow as tf

g = tf.Graph()
with g.as_default():
    A = tf.Variable(tf.random.normal([250, 160001]))
    B = tf.Variable(tf.random.normal([160001, 90]))
    #C = tf.matmul(A, B)
    flops = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print("flops: ", flops.total_float_ops)
    print("GFLOPs: ", flops.total_float_ops / 1000000000.0)
    print("trainable params: ", params.total_parameters)