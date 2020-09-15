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


if __name__ == '__main__':

    # TODO READ IN CONFIG 


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
        discrim_x_optimizer = ,
        discrim_y_optimizer = ,
        gen_g_optimizer = ,
        gen_f_optimizer = ,
        gen_loss_fn = ,
        discrim_loss_fn = 
    )