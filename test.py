# Model testing script - WIP
import matplotlib.pyplot as plt
import tensorflow as tf 
import yaml
import numpy as np 
from model import *

def decode_image(img):
    '''Decode jpg images and return 286,286,3 tensors'''

    img = tf.image.decode_jpeg(img, channels=3)

    return tf.image.resize(img, [286,286])   

def preprocess_test_image(img):
    '''
    Applies to test images
        - Resizes to [256,256,3]
        - Normalize to pixel range of [-1,1] as done in the original implementation
    '''
    # Resize

    img = tf.cast(img, dtype=tf.float32)
    return (img/127.5) - 1.0 # normalize

def load_test_image(filepath):
    '''
    Loads and preprocess test images
    '''

    img = tf.io.read_file(filepath)
    img = decode_image(img)
    img = preprocess_test_image(img)

    return(img)

if __name__ == '__main__':

    stream = open('model_config.yml', 'r')
    param_dict = yaml.load(stream, Loader=yaml.SafeLoader)

    # FILEPATHS
    PAINT_TRAIN_PATH = param_dict['PAINT_TRAIN_PATH']
    PHOTO_TRAIN_PATH = param_dict['PHOTO_TRAIN_PATH']
    PAINT_TEST_PATH = param_dict['PAINT_TEST_PATH']
    PHOTO_TEST_PATH = param_dict['PHOTO_TEST_PATH']
    INF_IMG_PATH  = param_dict['INF_IMG_PATH']

    FINAL_WEIGHTS_PATH = param_dict['FINAL_WEIGHTS_PATH']

    # MODEL PARAMETERS
    INPUT_SHAPE = param_dict['INPUT_SHAPE']
    DISCRIM_LR = param_dict['DISCRIM_LR']
    DISCRIM_BETA = param_dict['DISCRIM_BETA']
    GEN_LR = param_dict['GEN_LR']
    GEN_BETA = param_dict['GEN_BETA']
    K_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    GAMMA_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Load data  
    test_paintings = tf.data.Dataset.list_files(PAINT_TEST_PATH + '/*.jpg')
    test_photos = tf.data.Dataset.list_files(PHOTO_TEST_PATH + '/*.jpg')
    
    test_paintings = test_paintings.map(load_test_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)
    test_photos = test_photos.map(load_test_image, num_parallel_calls = AUTOTUNE).cache().shuffle(1000).batch(1)

    #Create model
    generator_g = build_generator(input_shape=INPUT_SHAPE, k_init = K_INIT, gamma_init = GAMMA_INIT)
    generator_f = build_generator(input_shape=INPUT_SHAPE, k_init = K_INIT, gamma_init = GAMMA_INIT)

    discriminator_x = build_discriminator(input_shape=INPUT_SHAPE, k_init= K_INIT, gamma_init = GAMMA_INIT)
    discriminator_y = build_discriminator(input_shape=INPUT_SHAPE, k_init=K_INIT,  gamma_init = GAMMA_INIT)

    c_gan_model = CycleGAN(discrim_x = discriminator_x, discrim_y = discriminator_y, 
                        gen_G = generator_g, gen_F = generator_f)
    
    c_gan_model.compile(
        discrim_x_optimizer = tf.keras.optimizers.Adam(learning_rate=DISCRIM_LR, beta_1=DISCRIM_BETA),
        discrim_y_optimizer = tf.keras.optimizers.Adam(learning_rate=DISCRIM_LR, beta_1=DISCRIM_BETA),
        gen_g_optimizer = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=GEN_BETA),
        gen_f_optimizer = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=GEN_BETA),
        gen_loss_fn = generator_loss ,
        discrim_loss_fn = discriminator_loss 
    )
    
    # Load in weights
    c_gan_model.load_weights(FINAL_WEIGHTS_PATH)

    # Plot and save some images
    _, ax = plt.subplots(4, 2, figsize=(10, 15))
    for i, img in enumerate(test_photos.take(4)):
    	prediction = c_gan_model.gen_G(img, training=False)[0].numpy()
    	prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    	img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    	ax[i, 0].imshow(img)
    	ax[i, 1].imshow(prediction)
    	ax[i, 0].set_title("Input image")
    	ax[i, 0].set_title("Input image")
    	ax[i, 1].set_title("Translated image")
    	ax[i, 0].axis("off")
    	ax[i, 1].axis("off")

    	prediction = tf.keras.preprocessing.image.array_to_img(prediction)
    	prediction.save(f"{INF_IMG_PATH}predicted_img_{i}.png")
    
    plt.tight_layout()
    plt.show()

