from keras.layers import Layer, Input, Dropout, Conv2D, Activation, add, UpSampling2D,     Conv2DTranspose, Flatten, Reshape
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import keras.backend as K
import tensorflow as tf
from skimage.transform import resize
from skimage import color
from helper_funcs import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ### Model parameters
# 
# This CycleGAN implementation allows a lot of freedom on both the training parameters and the network architecture.

opt = {}

# Data
opt['channels'] = 1
opt['img_shape'] = (200,200,1)


# CylceGAN can be used both on paired and unpaired data. The `paired_data` setting affects the presentation of output images as explained above.

opt['paired_data'] = False


# #### Training parameters
# - `lambda_ABA` and `lambda_BAB` set the importance of the cycle consistency losses in relation to the adversarial loss `lambda_adversarial`
# - `learning_rate_D` and `learning_rate_G` are the learning rates for the discriminators and generators respectively.
# - `generator_iterations` and `discriminator_iterations` represent how many times the generators or discriminators will be trained on every batch of images. This is very useful to keep the training of both systems balanced. In this case the discriminators become successful faster than the generators, so we account for this by training the generators 3 times on every batch of images.
# - `synthetic_pool_size` sets the size of the image pool used for training the discriminators. The image pool has a certain probability of returning a synthetic image from previous iterations, thus forcing the discriminator to have a certain "memory". More information on this method can be found in [this paper](https://arxiv.org/abs/1612.07828).
# - `beta_1` and `beta_2` are paremeters of the [Adam](https://arxiv.org/abs/1412.6980) optimizers used on the generators and discriminators.
# - `batch_size` determines the number of images used for each update of the network weights. Due to the significant memory requirements of CycleGAN it is difficult to use a large batch size. For the small example dataset values between 1-30 may be possible.
# - `epochs` sets the number of training epochs. Each epoch goes through all the training images once. The number of epochs necessary to train a model is therefore dependent on both the number of training images available and the batch size.

# Training parameters
opt['lambda_ABA'] = 10.0  # Cyclic loss weight A_2_B
opt['lambda_BAB'] = 10.0  # Cyclic loss weight B_2_A
opt['lambda_adversarial'] = 1.0  # Weight for loss from discriminator guess on synthetic images
opt['learning_rate_D'] = 2e-4
opt['learning_rate_G'] = 2e-4
opt['generator_iterations'] = 3  # Number of generator training iterations in each training loop
opt['discriminator_iterations'] = 1  # Number of discriminator training iterations in each training loop
opt['synthetic_pool_size'] = 50  # Size of image pools used for training the discriminators
opt['beta_1'] = 0.5  # Adam parameter
opt['beta_2'] = 0.999  # Adam parameter
opt['batch_size'] = 1  # Number of images per batch
opt['epochs'] = 10  # Choose multiples of 20 since the models are saved each 20th epoch


# Output parameters
opt['save_models'] = True  # Save or not the generator and discriminator models
opt['save_training_img'] = True  # Save or not example training results or only tmp.png
opt['save_training_img_interval'] = 1  # Number of epoch between saves of intermediate training results
opt['self.tmp_img_update_frequency'] = 3  # Number of batches between updates of tmp.png


# #### Architecture parameters
# - `use_instance_normalization` is supposed to allow the selection of instance normalization or batch normalization layes. At the moment only instance normalization is implemented, so this option does not do anything.
# - `use_dropout` and `use_bias` allows setting droupout layers in the generators and whether to use a bias term in the various convolutional layer in the genrators and discriminators.
# - `use_linear_decay` applies linear decay on the learning rates of the generators and discriminators,   `decay_epoch`
# - `use_patchgan` determines whether the discriminator evaluates the "realness" of images on a patch basis or on the whole. More information on PatchGAN can be found in [this paper](https://arxiv.org/abs/1611.07004).
# - `use_resize_convolution` provides two ways to perfrom the upsampling in the generator, with significant differences in the results. More information can be found in [this article](https://distill.pub/2016/deconv-checkerboard/). Each has its advantages, and we have managed to get successful result with both methods
# - `use_discriminator sigmoid` adds a sigmoid activation at the end of the discrimintator, forcing its output to the (0-1) range.

# Architecture parameters
opt['use_instance_normalization'] = True  # Use instance normalization or batch normalization
opt['use_dropout'] = False  # Dropout in residual blocks
opt['use_bias'] = True  # Use bias
opt['use_linear_decay'] = True  # Linear decay of learning rate, for both discriminators and generators
opt['decay_epoch'] = 101  # The epoch where the linear decay of the learning rates start
opt['use_patchgan'] = True  # PatchGAN - if false the discriminator learning rate should be decreased
opt['use_resize_convolution'] = True  # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
opt['discriminator_sigmoid'] = True  # Add a final sigmoid activation to the discriminator


# Tweaks
opt['REAL_LABEL'] = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss


# ### Model architecture
# 
# #### Layer blocks
# These are the individual layer blocks that are used to build the generators and discriminator. More information can be found in the appendix of the [CycleGAN paper](https://arxiv.org/abs/1703.10593).

# Discriminator layers
def ck(model, opt, x, k, use_normalization, use_bias):
    x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same', use_bias=use_bias)(x)
    print(x)
    if use_normalization:
        x = model['normalization'](axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)
    return x

# First generator layer
def c7Ak(model, opt, x, k):
    x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid', use_bias=opt['use_bias'])(x)
    x = model['normalization'](axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x

# Downsampling
def dk(model, opt, x, k):  # Should have reflection padding
    x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same', use_bias=opt['use_bias'])(x)
    x = model['normalization'](axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x

# Residual block
def Rk(model, opt, x0):
    k = int(x0.shape[-1])

    # First layer
    x = ReflectionPadding2D((1,1))(x0)
    x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=opt['use_bias'])(x)
    x = model['normalization'](axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)

    if opt['use_dropout']:
        x = Dropout(0.5)(x)

    # Second layer
    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=opt['use_bias'])(x)
    x = model['normalization'](axis=3, center=True, epsilon=1e-5)(x, training=True)
    # Merge
    x = add([x, x0])

    return x

# Upsampling
def uk(model, opt, x, k):
    # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
    if opt['use_resize_convolution']:
        x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=opt['use_bias'])(x)
    else:
        x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same', use_bias=opt['use_bias'])(x)  # this matches fractionally stided with stride 1/2
    x = model['normalization'](axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x


# #### Architecture functions


def build_generator(model, opt, name=None):
    # Layer 1: Input

    input_img = Input(shape=opt['img_shape'])
    x = ReflectionPadding2D((3, 3))(input_img)
    x = c7Ak(model, opt, x, 32)

    # Layer 2-3: Downsampling
    x = dk(model, opt, x, 64)
    x = dk(model, opt, x, 128)

    # Layers 4-12: Residual blocks
    for _ in range(4, 13):
        x = Rk(model, opt, x)

    # Layer 13:14: Upsampling
    x = uk(model, opt, x, 64)
    x = uk(model, opt, x, 32)

    # Layer 15: Output
    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(opt['channels'], kernel_size=7, strides=1, padding='valid', use_bias=True)(x)
    x = Activation('tanh')(x)
    # x = Reshape((217,181,1))(x)
    # print("Generator Model:")
    # print(Model(inputs=input_img, outputs=x, name=name).summary())
    return Model(inputs=input_img, outputs=x, name=name)


# #### Loss functions
# The discriminators use MSE loss. The generators use MSE for the adversarial losses and MAE for the cycle consistency losses.

# Mean squared error
def mse(y_true, y_pred):
    loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
    return loss

# Mean absolute error
def mae(y_true, y_pred):
    loss = tf.reduce_mean(tf.abs(y_pred - y_true))
    return loss


# Load Model
def load_model():
    model = {}
    # Normalization
    model['normalization'] = InstanceNormalization
    model['G_A2B'] = build_generator(model, opt, name='G_A2B_model')
    # Don't pre-allocate GPU memory; allocate as-needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))
    
    GA2B = model['G_A2B']
    
    GA2B.load_weights('saved_model/G_A2B_model_weights_epoch_200.hdf5')
    
    return GA2B



def predict(fname,model):
    image = mpimg.imread("images/"+fname)
    image = color.rgb2gray(image)
    image = resize(image,(200,200))
    print(image.shape)
    image = np.reshape(image,(1, 200,200,1))
    im = model.predict(image)
    im = np.reshape(im,(200,200))
    out_name = fname + '_result.png'
    out_dir ="results/" + out_name  
    mpimg.imsave(out_dir,im,cmap='gray')
    
    return out_name
