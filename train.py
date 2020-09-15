from model import *
import tensorflow as tf

###########################
# Image loading functions #
###########################

def decode_image(img):
    '''Decode jpg images and return 286,286,3 tensors'''

    img = tf.image.decode_jpg(img, channels=3)

    return tf.image.resize(img, [286,286]) 

def pre_process_train_image(img):
    '''
    Applies to training images:
        - Left Right random flip
        - Random Crop to [256,256,3]
        - Normalize to pixel range of [-1,1] as done in the original implementation
    '''
    # Random flip
    img = tf.image.random_flip_left_right(img)

    # Random crop
    img = tf.image.random_crop(img, size=[256,256,3])

    # Normalize to [-1,1]
    img = tf.cast(img, dtype=float32)
    return (img/127.5) - 1.0

def preprocess_test_image(img):
    '''
    Applies to test images
        - Resizes to [256,256,3]
        - Normalize to pixel range of [-1,1] as done in the original implementation
    '''

    # Resize
    img = tf.image.resize(img, [256,256])
    img = tf.cast(img, dtype=tf.float32)
    return (img/127.5) - 1.0

def load_train_image(filepath):
    '''
    Loads and preprocess training images
    '''

    img = tf.io.read_file(filepath)
    img = decode_image(img)
    img = preproces_train_image(img)

    return img

def load_test_image(filepath):
    '''
    Loads and preprocess test images
    '''

    img = tf.io.read_file(filepath)
    img = decode_image(img)
    img = preprocess_test_image(img)

    return(img)

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
        
        else:
            continue


if __name__ == '__main__':

    # TODO READ IN CONFIG 

    # FILEPATHS
    PAINT_TRAIN_PATH = 
    PHOTO_TRAIN_PATH = 
    PAINT_TEST_PATH = 
    PHOTO_TEST_PATH = 
    MONITOR_IMAGE_FILEPATH = 
    CHECKPOINT_FILEPATH = 
    FINAL_WEIGHTS_PATH = 

    # MODEL PARAMETERS
    INPUT_SHAPE = 
    K_INIT = 
    DISCRIM_OPT = 
    DISCRIM_LR = 
    GEN_OPT = 
    GEN_LR = 
    EPOCHS = 

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Create tensorflow datasets
    train_paintings = tf.data.Dataset.list_files(PAINT_TRAIN_PATH + '/*.jpg')
    train_photos = tf.data.Dataset.list_files(PHOTO_TRAIN_PATH + '/*.jpg')

    test_paintings = tf.data.Dataset.list_files(PAINT_TEST_PATH + '/*.jpg')
    test_photos = tf.data.Dataset.list_files(PHOTO_TEST_PATH + '/*.jpg')

    train_paintings = train_paintings.map(load_train_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)
    train_photos = train_photos.map(loat_train_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)

    test_paintings = test_paintings.map(load_test_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)
    test_photos = test_photos.map(load_test_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)


    # Create generators, discriminators and CycleGAN model

    generator_g = model.build_generator(input_shape=INPUT_SHAPE, k_init = K_INIT)
    generator_f = model.build_generator(input_shape=INPUT_SHAPE, k_init = K_INIT)

    discriminator_x = model.build_discriminator(input_shape=INPUT_SHAPE, k_init= K_INIT)
    discriminator_y = model.build_discriminator(input_shape=INPUT_SHAPE, k_init=K_INIT)

    c_gan_model = model.CycleGAN(discrim_x = discriminator_x, discrim_y = discriminator_y, gen_G = generator_g, gen_F = generator_F )

    # Compile model

    c_gan_model.compile(
        discrim_x_optimizer = DISCRIM_OPT,
        discrim_y_optimizer = DISCRIM_OPT,
        gen_g_optimizer = GEN_OPT ,
        gen_f_optimizer = GEN_OPT ,
        gen_loss_fn = generator_loss ,
        discrim_loss_fn = discriminator_loss 
    )

    # Set up Callbacks
    monitor = GANMonitor(MONITOR_IMAGE_FILEPATH)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = CHECKPOINT_FILEPATH, save_weights_only=True, save_format = SAVE FORMAT, save_freq=562*5
        ) #saves every 5 epochs

    # Fit
    c_gan_model.fit(
        tf.data.Dataset.zip((train_photos,train_paintings)),
        epochs = EPOCHS
        verbose = 1,
        callbacks = [monitor, ckpt_callback]
    )

    # Final model weights
    c_gan_mode.save_weights(FINAL_WEIGHTS_PATH, save_format=SAVE_FORMAT)