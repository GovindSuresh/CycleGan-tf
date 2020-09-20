# CGAN model file
# Contains the constituent parts that make up the model

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow_addons.layers import InstanceNormalization

class ReflectionPad2D(layers.Layer):
    def __init__(self, padding=(1,1)):
        self.padding = tuple(padding)
        super(ReflectionPad2D, self).__init__()

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [[0,0],[padding_height,padding_height],[padding_width,padding_width],[0,0]]

        return tf.pad(input_tensor,padding_tensor,mode='REFLECT')

class ResNetBlock(layers.Layer):
    def __init__(self, num_filters, kernel_init=None, use_1x1conv=False,strides=1):
        super(ResNetBlock, self).__init__()
        
        if kernel_init == None:
            self.kernel_init = tf.keras.initializers.RandomNormal(0.0,0.02) # Used in the original implementation
        else:
            self.kernel_init = kernel_init 
        
        self.conv_1 = layers.Conv2D(256, kernel_size=(3,3), strides=1, padding='valid', kernel_initializer = kernel_init, use_bias=False)
        self.conv_2 = layers.Conv2D(256, kernel_size=(3,3), strides=1, padding='valid', kernel_initializer = kernel_init, use_bias=False)
        self.conv_3 = None

        if use_1x1conv == True:
            self.conv_3 = layers.Conv2D(256, kernel_size=(1,1), strides=1)
        
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
        Y = self.instance_norm_2(self.conv_2(Y))

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
    x = ResNetBlock(256, kernel_init =k_init)(x)
    x = ResNetBlock(256, kernel_init =k_init)(x)
    x = ResNetBlock(256, kernel_init =k_init)(x)
    x = ResNetBlock(256, kernel_init =k_init)(x)
    x = ResNetBlock(256, kernel_init =k_init)(x)
    x = ResNetBlock(256, kernel_init =k_init)(x)
    x = ResNetBlock(256, kernel_init =k_init)(x)
    x = ResNetBlock(256, kernel_init =k_init)(x)
    x = ResNetBlock(256, kernel_init =k_init)(x)

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


def build_discriminator(input_shape, k_init):
    
    inp = layers.Input(shape=input_shape)
    
    #C64 block - No instance norm as per original implementation
    x = layers.Conv2D(64, kernel_size=(4,4), kernel_initializer=k_init, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #C128 block
    x = layers.Conv2D(128, kernel_size=(4,4), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.LeakyReLU(alpha=-0.2)(x)

    #C256 block
    x = layers.Conv2D(256, kernel_size=(4,4), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #C512 blocks
    x = layers.Conv2D(256, kernel_size=(4,4), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(512, kernel_size=(4,4), padding='same', kernel_initializer=k_init)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #Patch output based on PatchGAN
    output = layers.Conv2D(1, kernel_size=(4,4), padding='same', kernel_initializer=k_init)(x)
    output = InstanceNormalization(axis=-1)(output)
    output = layers.LeakyReLU(alpha=0.2)(output)

    return tf.keras.models.Model(inputs=inp, outputs=output)

class CycleGAN(tf.keras.Model):
    def __init__(self, discrim_x, discrim_y, gen_G, gen_F, lambda_val_cycle=10):
        super(CycleGAN, self).__init__()
        self.gen_G = gen_G
        self.gen_F = gen_F
        self.discrim_x = discrim_x
        self.discrim_y = discrim_y
        self.lambda_val_cycle = lambda_val_cycle 
        
    def compile(self, discrim_x_optimizer, discrim_y_optimizer, gen_g_optimizer, gen_f_optimizer, gen_loss_fn, discrim_loss_fn):
        super(CycleGAN, self).compile()
        
        self.discrim_x_optimizer = discrim_x_optimizer
        self.discrim_y_optimizer = discrim_y_optimizer
        self.gen_G_optimizer = gen_g_optimizer
        self.gen_F_optimizer = gen_f_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.discrim_loss_fn = discrim_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()
        
    def train_step(self, data):
        # X will be the real picss and y will be the paintings.
        real_x, real_y = data
        
        # CYCLEGAN TRAINING STEP 
        
        with tf.GradientTape(persistent=True) as tape:
            
            # Generate the fake images 
            generated_y = self.gen_G(real_x, training=True)
            generated_x = self.gen_F(real_y,training=True)
            
            # Generate the identity images
            identity_y = self.gen_G(real_y,training=True)
            identity_x = self.gen_F(real_x,training=True)
            
            # Generate the cycled images
            cycle_y = self.gen_G(generated_x,training=True)
            cycle_x = self.gen_F(generated_y,training=True)
            
            # Pass generated images into discriminator
            discrim_generated_x = self.discrim_x(generated_x,training=True)
            discrim_generated_y = self.discrim_y(generated_y,training=True)
            
            # Also pass real images into the discriminator
            discrim_real_x = self.discrim_x(real_x,training=True)
            discrim_real_y = self.discrim_y(real_y,training=True)
            
            ### CALCULATE LOSSES
            
            # Generator adversarial loss
            gen_G_loss = self.gen_loss_fn(discrim_generated_y)
            gen_F_loss = self.gen_loss_fn(discrim_generated_x)
            
            # Identity loss 
            identity_loss_G = self.identity_loss_fn(real_y, identity_y)
            identity_loss_F = self.identity_loss_fn(real_x, identity_x)
            
            # Cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, generated_y) * self.lambda_val_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, generated_x) * self.lambda_val_cycle
            
            # Total generator loss
            gen_G_total_loss = gen_G_loss + identity_loss_G + cycle_loss_G
            gen_F_total_loss = gen_F_loss + identity_loss_F + cycle_loss_F
                        
            # Discriminator_loss
            d_loss_x = self.discrim_loss_fn(discrim_real_x, discrim_generated_x)
            d_loss_y = self.discrim_loss_fn(discrim_real_y, discrim_generated_y)
            
            
        ### CALCULATE GRADIENTS
            
        # Generator Gradients using the tape.gradient() method
        gen_G_grads = tape.gradient(gen_G_total_loss, self.gen_G.trainable_variables)
        gen_F_grads = tape.gradient(gen_F_total_loss, self.gen_F.trainable_variables)

        # Discriminator Gradients
        discrim_x_grads = tape.gradient(d_loss_x, self.discrim_x.trainable_variables)
        discrim_y_grads = tape.gradient(d_loss_y, self.discrim_y.trainable_variables)

        # Update generator weights using the apply_gradients method of the optimizer
        self.gen_G_optimizer.apply_gradients(zip(gen_G_grads, self.gen_G.trainable_variables))
        self.gen_F_optimizer.apply_gradients(zip(gen_F_grads, self.gen_F.trainable_variables))

        # Update discriminator weights
        self.discrim_x_optimizer.apply_gradients(zip(discrim_x_grads, self.discrim_x.trainable_variables))
        self.discrim_y_optimizer.apply_gradients(zip(discrim_y_grads, self.discrim_y.trainable_variables))
        
        #train step function updates the weights in mem and returns the loss. 
        return {
            'gen_G_loss': gen_G_total_loss,
            'gen_F_loss': gen_F_total_loss,
            'discrim_x_loss': d_loss_x,
            'discrim_y_loss': d_loss_y,
        }