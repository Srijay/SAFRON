# example of defining a u-net encoder-decoder generator model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LeakyReLU
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
import numpy as np
import tempfile
from tensorflow.python.keras import Sequential
import os

def count_flops(model):
    """ Count flops of a keras model
    # Args.
        model: Model,
    # Returns
        int, FLOPs of a model
    # Raises
        TypeError, if a model is not an instance of Sequence or Model
    """

    if not isinstance(model, (Sequential, Model)):
        raise TypeError(
            'Model is expected to be an instance of Sequential or Model, '
            'but got %s' % type(model))

    output_op_names = [_out_tensor.op.name for _out_tensor in model.outputs]

    sess = tf.keras.backend.get_session()
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), output_op_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(os.path.join(tmpdir, 'graph.pb'))
        with tf.gfile.GFile(graph_file, "wb") as f:
            f.write(frozen_graph_def.SerializeToString())

        with tf.gfile.GFile(graph_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as new_graph:
            tf.import_graph_def(graph_def, name='')
            tfprof_opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(new_graph, options=tfprof_opts)

    return flops

#input_img = tf.keras.Input(batch_shape=(1, 224, 224, 3))

model = tf.keras.applications.InceptionV3(
        include_top=True, weights=None,
        input_tensor=tf.keras.Input(batch_shape=(1, 224, 224, 3)))
model.summary()

#flops = count_flops(model)

flops = tf.profiler.profile(K.get_session().graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
#params = tf.profiler.profile(K.get_session().graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
print("flops: ", flops.total_float_ops)
print("GFLOPs: ", flops.total_float_ops / 1000000000.0)
#print("trainable params: ", params.total_parameters)