# example of defining a u-net encoder-decoder generator model
from tensorflow.keras.initializers import RandomNormal
# example of defining a u-net encoder-decoder generator model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LeakyReLU,Conv2DTranspose,Activation,Concatenate
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
import tempfile
from tensorflow.python.keras import Sequential
import os
from tensorflow.keras.optimizers import Adam
import numpy as np
import time

#######################CORRECT VERSION#################

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
            writer = tf.summary.FileWriter('gg', graph=new_graph)
            writer.flush()

    return flops

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the discriminator model
def define_discriminator(im_size):
    # weight initialization
    in_src_image = tf.keras.Input(batch_shape=(1, im_size, im_size, 3))
    in_target_image = tf.keras.Input(batch_shape=(1, im_size, im_size, 3))

    init = RandomNormal(stddev=0.02)
    # source image input
    #in_src_image = Input(shape=input_src_img)
    # target image input
    #in_target_image = Input(shape=input_src_tgt)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C1024
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C2048
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C4096
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C8192
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

# define the standalone generator model
def define_generator(im_size):
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = tf.keras.Input(batch_shape=(1, im_size, im_size, 3))

    #g = Conv2D(64, (41, 41), strides=(1, 1), padding='valid', kernel_initializer=init)(in_image)
    # conditionally add batch normalization
    # leaky relu activation
    #g = LeakyReLU(alpha=0.2)(g)

    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    e8 = define_encoder_block(e7, 512)  # for size 512
    e9 = define_encoder_block(e8, 512)  # for size 1024
    e10 = define_encoder_block(e9, 512)  # for size 2048
    e11 = define_encoder_block(e10, 512)  # for size 4096
    #e12 = define_encoder_block(e11, 512)  # for size 8192
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e11)
    b = Activation('relu')(b)
    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    #d_4 = decoder_block(b, e12, 512)
    d_3 = decoder_block(b, e11, 512)
    d_2 = decoder_block(d_3, e10, 512)
    d_1 = decoder_block(d_2, e9, 512)
    d0 = decoder_block(d_1, e8, 512)  # for size 512
    d1 = decoder_block(d0, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	# d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model


def get_model_memory_usage(batch_size, model):

    from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


# create the model
imsize = 4096
g_model = define_generator(imsize)
d_model = define_discriminator(imsize)

image_shape = (imsize,imsize,3)
#gan_model = define_gan(g_model, d_model, image_shape)

# print(get_model_memory_usage(1,d_model))
# print(get_model_memory_usage(1,g_model))
# exit(0)

# summarize the model
#gan_model.summary()
#exit(0)

flops = count_flops(g_model)

#flops = tf.profiler.profile(K.get_session().graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
params = tf.profiler.profile(K.get_session().graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
print("flops: ", flops.total_float_ops)
print("GFLOPs: ", flops.total_float_ops / 1000000000.0)
print("trainable params: ", params.total_parameters)


