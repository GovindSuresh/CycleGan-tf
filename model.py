# CGAN model file
# Contains the constituent parts that make up the model

import tensorflow as tf
import tensorflow.keras.layers as layers
from tfa.layers import InstanceNormalization

class ReflectionPad2D(layers.Layer):
    def __init__(self, padding=(1,1)):
        self.padding = tuple(padding)
        super(ReflectionPad2D, self).__init__()

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [[0,0],[padding_height,padding_height],[padding_width,padding_width],[0,0],[0,0],]

        return tf.pad(input_tensor,padding_tensor,mode='REFLECT')



class ResNetBlock(layers.Layer):
    def __init__(self, num_filters, kernel_init, use_1x1conv=False,strides=1):
        super(ResNetBlock, self).__init__()
        
        if kernel_init = None:
            self.kernel_init = tf.keras.initializers.RandomNormal(0.0,0.02) # Used in the original implementation
        else:
            self.kernel_init = kernel_init 
        
        self.conv_1 = layers.Conv2D(256, kernel_size=(3,3), strides=1, padding='same', kernel_initializer = init)
        self.conv_2 = layers.Conv2D(256, kernel_size=(3,3), strides=1, padding='same', kernel_initializer = init)
        self.conv_3 = None

        if use_1x1conv == True:
            self.conv_3 = layers.Conv2D(256, kernel_size=(1,1). strides=1)
        
        # Normalization layers
        self.instance_norm_1 = InstanceNormalization(axis=-1)
        self.instance_norm_2 = InstanceNormalization(axis=-1)

        # Reflection padding layers
        self.reflect_pad1 = ReflectionPad2D()
        self.reflect_pad2 = ReflectionPad2D()

    def call(self, X):
        # Reflection pad -> Conv -> Instance Norm -> Relu -> Reflection pad -> conv -> Instance Norm -> concat output and input
        
        Y = self.reflect_pad1(X)
        Y =  tf.keras.activations.relu(self.instance_norm_1(self.conv_1(X)))
        Y = self.reflect_pad2(X)
        Y = self.instance_norm_2(self.conv_2(Y)))

        Y = tf.add(Y,X)

        return Y

def build_generator(input_shape, k_init):
    
    inp = layers.Input(shape=input_shape)
    # InstanceNorm
    x = layers.Conv2D(64,kernel_size=(7,7), kernel_initializer=k_init, strides=1, padding='same')(inp)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    #Downsampling Layers
    x = layers.Conv2D(128, kernel_size=(3,3), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, kernel_size=(3,3), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    #ResNet Blocks
    x = ResNetBlock(256)(x)
    x = ResNetBlock(256)(x)
    x = ResNetBlock(256)(x)
    x = ResNetBlock(256)(x)
    x = ResNetBlock(256)(x)
    x = ResNetBlock(256)(x)
    x = ResNetBlock(256)(x)
    x = ResNetBlock(256)(x)
    x = ResNetBlock(256)(x)

    #Upsampling layers
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3,3), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3,3), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    # Final block 
    last_layer = tf.keras.layers.Conv2DTranspose(3, kernel_size=(7,7), kernel_initializer=k_init, strides=1, padding='same')(x)
    last_layer = InstanceNormalization(axis=-1)(last_layer)
    # as with the original paper, the last activation is tanh rather than relu 
    last_layer = layers.Activation('tanh')(last_layer)
    
    return tf.keras.models.Model(inputs=inp, outputs=last_layer)
