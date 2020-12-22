# Model testing script - WIP

import yaml
from model import *
from data_loading import *

if __name__ == '__main__':

    stream = open('model_config.yml', 'r')
    param_dict = yaml.load(stream, Loader=yaml.SafeLoader)

    # FILEPATHS
    PAINT_TRAIN_PATH = param_dict['PAINT_TRAIN_PATH']
    PHOTO_TRAIN_PATH = param_dict['PHOTO_TRAIN_PATH']
    PAINT_TEST_PATH = param_dict['PAINT_TEST_PATH']
    PHOTO_TEST_PATH = param_dict['PHOTO_TEST_PATH']
    FINAL_WEIGHTS_PATH = param_dict['FINAL_WEIGHTS_PATH']

    #Create model

    generator_g = build_generator(input_shape=INPUT_SHAPE, k_init = K_INIT)
    generator_f = build_generator(input_shape=INPUT_SHAPE, k_init = K_INIT)

    discriminator_x = build_discriminator(input_shape=INPUT_SHAPE, k_init= K_INIT)
    discriminator_y = build_discriminator(input_shape=INPUT_SHAPE, k_init=K_INIT)

    c_gan_model = CycleGAN(discrim_x = discriminator_x, discrim_y = discriminator_y, 
                        gen_G = generator_g, gen_F = generator_f)

    
    # Load in weights
    c_gan_model.load_weights(FINAL_WEIGHTS_PATH)




