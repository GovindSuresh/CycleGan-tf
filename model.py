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
        padding_tensor = [[0,0], [padding_height, padding_height],[padding_width, padding_width], [0,0],]

        return tf.pad(input_tensor, padding_tensor, mode='REFLECT')

class ResNetBlock(layers.Layer):
    def __init__(self, num_filters, kernel_init=None, gamma_init=None, use_1x1conv=False, strides=1):
        super(ResNetBlock, self).__init__()
        
        if kernel_init == None:
            self.kernel_init = tf.keras.initializers.RandomNormal(0.0,0.02) # Used in the original implementation
        else:
            self.kernel_init = kernel_init
        
        if gamma_init == None:
            self.gamma_init = tf.keras.initializers.RandomNormal(0.0,0.02) # Used in the original implementation
        else:
            self.kernel_init = kernel_init  
        
        self.conv_1 = layers.Conv2D(256, kernel_size=(3,3), strides=(1, 1), padding='valid', 
                                    kernel_initializer = self.kernel_init, use_bias=False)
        self.conv_2 = layers.Conv2D(256, kernel_size=(3,3), strides=(1, 1), padding='valid', 
                                    kernel_initializer = self.kernel_init, use_bias=False)
        self.conv_3 = None

        if use_1x1conv == True:
            self.conv_3 = layers.Conv2D(256, kernel_size=(1,1), strides=1)
        
        # Normalization layers
        self.instance_norm_1 = InstanceNormalization(axis=-1, gamma_initializer = self.gamma_init)
        self.instance_norm_2 = InstanceNormalization(axis=-1, gamma_initializer = self.gamma_init)

        # Reflection padding layers
        self.reflect_pad1 = ReflectionPad2D()
        self.reflect_pad2 = ReflectionPad2D()

    def call(self, X):
        # Reflection pad -> Conv -> Instance Norm -> Relu -> Reflection pad -> conv -> 
        # Instance Norm -> concat output and input
        
        Y = self.reflect_pad1(X)
        Y =  tf.keras.activations.relu(self.instance_norm_1(self.conv_1(Y), training=True))
        Y = self.reflect_pad2(Y)
        Y = self.instance_norm_2(self.conv_2(Y), training=True)

        Y = layers.add([X,Y])

        return Y

def build_generator(input_shape, k_init, gamma_init):
    
    inp = layers.Input(shape=input_shape)
    
    x = ReflectionPad2D(padding=(3,3))(inp)
    x = layers.Conv2D(64,kernel_size=(7,7), kernel_initializer=k_init, strides=1, padding='same',
                    use_bias=False)(inp)
    x = InstanceNormalization(axis=-1, gamma_initializer = gamma_init)(x)
    x = layers.Activation('relu')(x)

    #Downsampling Layers
    x = layers.Conv2D(128, kernel_size=(3,3), kernel_initializer=k_init, strides=2, padding='same',
                    use_bias=False)(x)
    x = InstanceNormalization(axis=-1, gamma_initializer = gamma_init)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, kernel_size=(3,3), kernel_initializer=k_init, strides=2, padding='same',
                    use_bias=False)(x)
    x = InstanceNormalization(axis=-1, gamma_initializer = gamma_init)(x)
    x = layers.Activation('relu')(x)

    #ResNet Blocks
    x = ResNetBlock(256, kernel_init=k_init)(x)
    x = ResNetBlock(256, kernel_init=k_init)(x)
    x = ResNetBlock(256, kernel_init=k_init)(x)
    x = ResNetBlock(256, kernel_init=k_init)(x)
    x = ResNetBlock(256, kernel_init=k_init)(x)
    x = ResNetBlock(256, kernel_init=k_init)(x)
    x = ResNetBlock(256, kernel_init=k_init)(x)
    x = ResNetBlock(256, kernel_init=k_init)(x)
    x = ResNetBlock(256, kernel_init=k_init)(x)

    #Upsampling layers
    x = layers.Conv2DTranspose(128, kernel_size=(3,3), kernel_initializer=k_init, strides=(2,2), padding='same',
                            use_bias=False)(x)
    x = InstanceNormalization(axis=-1, gamma_initializer = gamma_init)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(64, kernel_size=(3,3), kernel_initializer=k_init, strides=(2,2), padding='same',
                            use_bias=False)(x)
    x = InstanceNormalization(axis=-1, gamma_initializer = gamma_init)(x)
    x = layers.Activation('relu')(x)

    # Final block 
    last_layer = ReflectionPad2D(padding=(3,3))(x)
    last_layer = layers.Conv2D(3, kernel_size=(7,7), padding='valid')(last_layer)
    
    # as with the original paper, the last activation is tanh rather than relu 
    last_layer = layers.Activation('tanh')(last_layer)
    
    return tf.keras.models.Model(inputs=inp, outputs=last_layer)


def build_discriminator(input_shape, k_init, gamma_init):
    
    inp = layers.Input(shape=input_shape)
    
    #C64 block - No instance norm as per original implementation
    x = layers.Conv2D(64, kernel_size=(4,4), kernel_initializer=k_init, strides=(2, 2), padding='same')(inp)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #C128 block
    x = layers.Conv2D(128, kernel_size=(4,4), kernel_initializer=k_init, strides=(2,2), padding='same',
                    use_bias=False)(x)
    x = InstanceNormalization(axis=-1, gamma_initializer = gamma_init)(x)
    x = layers.LeakyReLU(alpha=-0.2)(x)

    #C256 block
    x = layers.Conv2D(256, kernel_size=(4,4), kernel_initializer=k_init, strides=2, padding='same',
                    use_bias=False)(x)
    x = InstanceNormalization(axis=-1, gamma_initializer = gamma_init)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #C512 block
    x = layers.Conv2D(512, kernel_size=(4,4), padding='same', kernel_initializer=k_init,
                    use_bias=False)(x)
    x = InstanceNormalization(axis=-1, gamma_initializer = gamma_init)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    #Patch output based on PatchGAN
    output = layers.Conv2D(1, kernel_size=(4,4),strides=(1,1), padding='same', kernel_initializer=k_init)(x)

    return tf.keras.models.Model(inputs=inp, outputs=output)

def discriminator_loss(discrim_real, discrim_generated):
    '''
    Inputs: 
    discrim_real - The real image
    discrim_generated - The generated image from the generator
    
    Outputs:
    total_discrim_loss - The sum of the real_loss and generated loss
    
    The function calculates the real and generated losses as the MSE between the discriminator output and the targets 
    y_true is a matrix of 1's for the real images and 0s for the generated images.
    '''
    loss_object_mse = tf.keras.losses.MeanSquaredError()
    real_loss = loss_object_mse(tf.ones_like(discrim_real), discrim_real)
    generated_loss = loss_object_mse(tf.zeros_like(discrim_generated), discrim_generated)
    total_discrim_loss = (real_loss + generated_loss) * 0.5 # multiply by 0.5 as suggested by authors
    
    return total_discrim_loss

def generator_loss(discrim_gen_output):
    
    # If the discriminator thinks the generated output is real (1) 
    # then the MSE between a matrix of ones and the discrim output will be smaller
    loss_object_mse = tf.keras.losses.MeanSquaredError()
    gen_loss = loss_object_mse(tf.ones_like(discrim_gen_output), discrim_gen_output)
    
    return gen_loss

class CycleGAN(tf.keras.Model):
    def __init__(self, discrim_x, discrim_y, gen_G, gen_F, lambda_val_cycle=10, lambda_val_identity=0.5):
        super(CycleGAN, self).__init__()
        self.gen_G = gen_G
        self.gen_F = gen_F
        self.discrim_x = discrim_x
        self.discrim_y = discrim_y
        self.lambda_val_cycle = lambda_val_cycle
        self.lambda_val_identity = lambda_val_identity 
        
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
        # X will be the real pics and y will be the paintings.
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
            identity_loss_G = self.identity_loss_fn(real_y, identity_y) * self.lambda_val_cycle * self.lambda_val_identity
            identity_loss_F = self.identity_loss_fn(real_x, identity_x) * self.lambda_val_cycle * self.lambda_val_identity
            
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

## CUSTOM CALLBACKS
# Gan Monitor callback to print out images during training - taken from https://keras.io/examples/generative/cyclegan/

class GANMonitor(tf.keras.callbacks.Callback):
    '''
    Callback to generate and save images after each epoch
    Taken from https://keras.io/examples/generative/cyclegan/
    '''
    
    def __init__(self, monitor_image_filepath, num_img=4):
        self.monitor_image_filepath = monitor_image_filepath
        self.num_img = num_img
        
    
    def on_epoch_end(self, epoch, logs=None):
        
        # Generate 4 images, show on screen and save to file every 5 epochs
        if epoch % 5 == 0:
        
            _, ax = plt.subplots(4,2,figsize=(12,12))
            
            for i, img in enumerate(train_photos.take(4)):
                prediction = self.model.gen_G(img)[0].numpy()
                prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
                img = (img[0] * 127.5 +127.5).numpy().astype(np.uint8)
                
                ax[i, 0].imshow(img)
                ax[i, 1].imshow(prediction)
                ax[i, 0].set_title("Input image")
                ax[i, 1].set_title("Translated image")
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")
                
                prediction = tf.keras.preprocessing.image.array_to_img(prediction)
                prediction.save(
                    "{monitor_image_filepath}/generated_img_{i}_{epoch}.png".format(monitor_image_filepath = self.monitor_image_filepath, i=i, epoch=epoch+1) 
                                )
            plt.show()
            plt.close()
        
        