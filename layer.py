from Dropblock import *
from keras.layers import *
def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True,keep_prob=0.5):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = Dropout(keep_prob)(x)
    if activation:
        x = BatchActivate(x)
    return x
def residual_block(blockInput, num_filters=16, batch_activate=False,keep_prob=0.5):

    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3, 3),keep_prob=keep_prob)
    x = convolution_block(x, num_filters, (3, 3), activation=False,keep_prob=keep_prob)
    if blockInput.get_shape().as_list()[-1] !=  x.get_shape().as_list()[-1]:
        blockInput = Conv2D(num_filters, (1, 1), activation=None, padding="same")(blockInput)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x
def residual_drop_block(blockInput, num_filters=16, batch_activate=False,keep_prob=0.9,block_size=7):
    x = BatchActivate(blockInput)
    x = convolution_block_dropblock(x, num_filters, (3, 3),keep_prob=keep_prob,block_size=block_size)
    x = convolution_block_dropblock(x, num_filters, (3, 3), activation=False,keep_prob=keep_prob,block_size=block_size)
    if blockInput.get_shape().as_list()[-1] !=  x.get_shape().as_list()[-1]:
        blockInput = Conv2D(num_filters, (1, 1), activation=None, padding="same")(blockInput)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x
def convolution_block_dropblock(x, filters, size, strides=(1, 1), padding='same', activation=True,keep_prob=0.9,block_size=7):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = DropBlock2D(block_size=block_size,keep_prob=keep_prob)(x)
    if activation:
        x = BatchActivate(x)
    return x