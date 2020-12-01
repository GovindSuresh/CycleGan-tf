from model import *
import tensorflow as tf
import yaml

###########################
# Image loading functions #
###########################

def decode_image(img):
    '''Decode jpg images and return 286,286,3 tensors'''

    img = tf.image.decode_jpeg(img, channels=3)

    return tf.image.resize(img, [286,286]) 

def preprocess_train_image(img):
    '''
    Applies to training images:
        - Left Right random flip
        - Random Crop to [256,256,3]
        - Normalize to pixel range of [-1,1] as done in the original implementation
    '''
    # Random flip
    img = tf.image.random_flip_left_right(img)

    # Random crop
    img = tf.image.random_crop(img, size=INPUT_SHAPE)

    # Normalize to [-1,1]
    img = tf.cast(img, dtype=tf.float32)
    return (img/127.5) - 1.0

def preprocess_test_image(img):
    '''
    Applies to test images
        - Resizes to [256,256,3]
        - Normalize to pixel range of [-1,1] as done in the original implementation
    '''

    # Resize
    img = tf.image.resize(img, INPUT_SHAPE[:-1])
    img = tf.cast(img, dtype=tf.float32)
    return (img/127.5) - 1.0

def load_train_image(filepath):
    '''
    Loads and preprocess training images
    '''

    img = tf.io.read_file(filepath)
    img = decode_image(img)
    img = preprocess_train_image(img)

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
        
        


if __name__ == '__main__':

    # Check how many GPU's available
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    # READ IN CONFIG
    stream = open('model_config.yml', 'r')
    param_dict = yaml.load(stream, Loader=yaml.SafeLoader)

    # FILEPATHS
    PAINT_TRAIN_PATH = param_dict['PAINT_TRAIN_PATH']
    PHOTO_TRAIN_PATH = param_dict['PHOTO_TRAIN_PATH']
    PAINT_TEST_PATH = param_dict['PAINT_TEST_PATH']
    PHOTO_TEST_PATH = param_dict['PHOTO_TEST_PATH']
    MONITOR_IMAGE_FILEPATH = param_dict['MONITOR_IMAGE_FILEPATH']
    CHECKPOINT_FILEPATH = param_dict['CHECKPOINT_FILEPATH']
    FINAL_WEIGHTS_PATH = param_dict['FINAL_WEIGHTS_PATH']

    # MODEL PARAMETERS
    INPUT_SHAPE = param_dict['INPUT_SHAPE']
    DISCRIM_LR = param_dict['DISCRIM_LR']
    DISCRIM_BETA = param_dict['DISCRIM_BETA']
    GEN_LR = param_dict['GEN_LR']
    GEN_BETA = param_dict['GEN_BETA']
    EPOCHS = param_dict['EPOCHS']
    K_INIT = tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02)

    # CHECKPOINT PARAMETERS
    MONITOR = param_dict['MONITOR']
    CHECKPOINTS = param_dict['CKPTS']
    
    # GENERAL
    SAVE_FORMAT = param_dict['SAVE_FORMAT']
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Create tensorflow datasets
    train_paintings = tf.data.Dataset.list_files(PAINT_TRAIN_PATH + '/*.jpg')
    train_photos = tf.data.Dataset.list_files(PHOTO_TRAIN_PATH + '/*.jpg')

    test_paintings = tf.data.Dataset.list_files(PAINT_TEST_PATH + '/*.jpg')
    test_photos = tf.data.Dataset.list_files(PHOTO_TEST_PATH + '/*.jpg')

    train_paintings = train_paintings.map(load_train_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)
    train_photos = train_photos.map(load_train_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)

    test_paintings = test_paintings.map(load_test_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)
    test_photos = test_photos.map(load_test_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)


    # Create generators, discriminators and CycleGAN model

    generator_g = build_generator(input_shape=INPUT_SHAPE, k_init = K_INIT)
    generator_f = build_generator(input_shape=INPUT_SHAPE, k_init = K_INIT)

    discriminator_x = build_discriminator(input_shape=INPUT_SHAPE, k_init= K_INIT)
    discriminator_y = build_discriminator(input_shape=INPUT_SHAPE, k_init=K_INIT)

    c_gan_model = CycleGAN(discrim_x = discriminator_x, discrim_y = discriminator_y, 
                        gen_G = generator_g, gen_F = generator_f)

    # Compile model

    c_gan_model.compile(
        discrim_x_optimizer = tf.keras.optimizers.Adam(learning_rate=DISCRIM_LR, beta_1=DISCRIM_BETA),
        discrim_y_optimizer = tf.keras.optimizers.Adam(learning_rate=DISCRIM_LR, beta_1=DISCRIM_BETA),
        gen_g_optimizer = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=GEN_BETA),
        gen_f_optimizer = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=GEN_BETA),
        gen_loss_fn = generator_loss ,
        discrim_loss_fn = discriminator_loss 
    )

    # Set up Callbacks
    callback_list = []
    if MONITOR == True:
        monitor = GANMonitor(MONITOR_IMAGE_FILEPATH)
        callback_list.append(monitor)
    
    if CHECKPOINTS == True: 
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = CHECKPOINT_FILEPATH, save_weights_only=True, save_format = SAVE_FORMAT, save_freq=562*5) #saves every 5 epochs
        callback_list.append(ckpt_callback)
    
    # Fit
    c_gan_model.fit(
        tf.data.Dataset.zip((train_photos,train_paintings)),
        epochs = EPOCHS,
        verbose = 1,
        callbacks = callback_list
    )

    # Final model weights
    c_gan_mode.save_weights(FINAL_WEIGHTS_PATH, save_format=SAVE_FORMAT)