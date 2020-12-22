import tensorflow as tf
import yaml
from model import *
from data_loading import *

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


    #GENERAL
    SAVE_FORMAT = param_dict['SAVE_FORMAT']
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Create tensorflow datasets
    train_paintings = tf.data.Dataset.list_files(PAINT_TRAIN_PATH + '/*.jpg')
    train_photos = tf.data.Dataset.list_files(PHOTO_TRAIN_PATH + '/*.jpg')

    test_paintings = tf.data.Dataset.list_files(PAINT_TEST_PATH + '/*.jpg')
    test_photos = tf.data.Dataset.list_files(PHOTO_TEST_PATH + '/*.jpg')

    train_paintings = train_paintings.map(load_train_image(*,size=INPUT_SHAPE), num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)
    train_photos = train_photos.map(load_train_image(*,size=INPUT_SHAPE), num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)

    test_paintings = test_paintings.map(load_test_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)
    test_photos = test_photos.map(load_test_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)


    # Create generators, discriminators and CycleGAN model

    generator_g = build_generator(input_shape=INPUT_SHAPE, k_init = K_INIT)
    generator_f = build_generator(input_shape=INPUT_SHAPE, k_init = K_INIT)

    discriminator_x = build_discriminator(input_shape=INPUT_SHAPE, k_init= K_INIT)
    discriminator_y = build_discriminator(input_shape=INPUT_SHAPE, k_init=K_INIT)

    c_gan_model = CycleGAN(discrim_x = discriminator_x, discrim_y = discriminator_y, gen_G = generator_g, gen_F = generator_f )

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
<<<<<<< HEAD
    callback_list = []
    if MONITOR == True:
        monitor = GANMonitor(MONITOR_IMAGE_FILEPATH)
        callback_list.append(monitor)
    
    if CHECKPOINTS == True: 
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = CHECKPOINT_FILEPATH, save_weights_only=True, save_format = SAVE_FORMAT, save_freq=562*5) #saves every 5 epochs
        callback_list.append(ckpt_callback)

    if TENSORBOARD == True:
        log_dir = LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callback_list.append(tensorboard)
    
=======
    monitor = GANMonitor(MONITOR_IMAGE_FILEPATH)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = CHECKPOINT_FILEPATH, save_weights_only=True, save_format = SAVE_FORMAT, save_freq=562*5) #saves every 5 epochs

>>>>>>> parent of 405e253... added options to choose callbacks
    # Fit
    c_gan_model.fit(
        tf.data.Dataset.zip((train_photos,train_paintings)),
        epochs = EPOCHS,
        verbose = 1,
        callbacks = [monitor, ckpt_callback]
    )

    # Final model weights
    c_gan_model.save_weights(FINAL_WEIGHTS_PATH, save_format=SAVE_FORMAT)