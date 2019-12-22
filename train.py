# now load attribute

# 1.A.2

import sys
import os
args = sys.argv

model_file_path = "models"
model_file_name = "vae_model_70_20_10"
if len(args) >= 2:
    model_file_name = args[1]

import pandas as pd
attr = pd.read_csv('data/list_attr_celeba.csv')
attr = attr.set_index('image_id')

# check if attribute successful loaded
attr.describe()

import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from data_generator import auto_encoder_generator, get_input

IMG_NAME_LENGTH = 6
file_path = "data/img_align_celeba/"
img_id = np.arange(1,len(attr.index)+1)
img_path = []
for i in range(len(img_id)):
    img_path.append(file_path + (IMG_NAME_LENGTH - len(str(img_id[i])))*'0' + str(img_id[i]) + '.jpg')

# pick 80% as training set and 20% as validation set
train_path = img_path[:int((0.7)*len(img_path))]
val_path = img_path[int((0.7)*len(img_path)):int((0.7)*len(img_path))+int((0.2)*len(img_path))]
test_path = img_path[int((0.7)*len(img_path))+int((0.2)*len(img_path)):]

print(train_path[0], train_path[-1])
print(val_path[0], val_path[-1])
print(test_path[0], test_path[-1])

pickle.dump({"train":train_path, "val":val_path, "test":test_path}, open("train_val_test_split.pkl", 'wb'))

train_generator = auto_encoder_generator(train_path,32)
val_generator = auto_encoder_generator(val_path,32)

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    ax[i].imshow(get_input(img_path[i+3]))
    ax[i].axis('off')
    ax[i].set_title(img_path[i+3][-10:])
plt.show()

print(attr.iloc[:3])

b_size = 32
n_size = 512

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D, Cropping2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from utils import sampling, vae_loss

def build_conv_vae(input_shape, bottleneck_size, sampling, batch_size = 32):

    # ENCODER
    input = Input(shape=(input_shape[0],input_shape[1],input_shape[2]), name='myInput')
    x = Conv2D(32,(3,3),activation = 'relu', padding = 'same')(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding ='same')(x)
    x = Conv2D(64,(3,3),activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding ='same')(x)
    x = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding ='same')(x)
    x = Conv2D(256,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding ='same')(x)

    # Latent Variable Calculation
    shape = K.int_shape(x)
    flatten_1 = Flatten()(x)
    dense_1 = Dense(bottleneck_size, name='z_mean')(flatten_1)
    z_mean = BatchNormalization()(dense_1)
    flatten_2 = Flatten()(x)
    dense_2 = Dense(bottleneck_size, name ='z_log_sigma')(flatten_2)
    z_log_sigma = BatchNormalization()(dense_2)
    z = Lambda(sampling)([z_mean, z_log_sigma])
    encoder = Model(input, [z_mean, z_log_sigma, z], name = 'encoder')
    latent_input = Input(shape=(bottleneck_size,), name = 'decoder_input')
    x = Dense(shape[1]*shape[2]*shape[3])(latent_input)
    x = Reshape((shape[1],shape[2],shape[3]))(x)
    x = UpSampling2D((2,2))(x)
    x = Cropping2D([[0,0],[0,1]])(x)
    x = Conv2DTranspose(256,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Cropping2D([[0,1],[0,1]])(x)
    x = Conv2DTranspose(128,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Cropping2D([[0,1],[0,1]])(x)
    x = Conv2DTranspose(64,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(32,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    output = Conv2DTranspose(3,(3,3), activation = 'tanh', padding ='same', name='myOutput')(x)
    decoder = Model(latent_input, output, name = 'decoder')

    output_2 = decoder(encoder(input)[2])
    print(output_2)
    vae = Model(input, output_2, name ='vae')
    return vae, input, output, encoder, decoder, z_mean, z_log_sigma

img_sample = get_input(img_path[0])
print(img_sample.shape)
vae_2, input, output, encoder, decoder, z_mean, z_log_sigma = build_conv_vae(img_sample.shape, n_size, sampling, batch_size = b_size)

import os
from tensorflow.keras.models import load_model
if os.path.exists('models/vae_model_70_20_10'):
    loss = vae_loss(z_mean, z_log_sigma)
    vae_2 = load_model('models/vae_model_70_20_10', custom_objects={'sampling':sampling,'loss':loss})
    vae_2.summary()
    import random
    x_test = []
    for i in range(64):
        x_test.append(get_input(img_path[random.randint(0,len(img_id))]))
    x_test = np.array(x_test)
    figure_Decoded = vae_2.predict(x_test.astype('float32')/127.5 -1, batch_size = b_size)
    figure_original = x_test[0]
    figure_decoded = (figure_Decoded[0]+1)/2
    for i in range(4):
        plt.axis('off')
        plt.subplot(2,4,1+i*2)
        plt.imshow(x_test[i])
        plt.axis('off')
        plt.subplot(2,4,2 + i*2)
        plt.imshow((figure_Decoded[i]+1)/2)
        plt.axis('off')
    plt.show()
else:
    print("encoder summary:")
    encoder.summary()
    print("decoder summary:")
    decoder.summary()
    print("vae summary:")
    vae_2.summary()
    
    loss = vae_loss(z_mean, z_log_sigma)
    vae_2.compile(optimizer='rmsprop', loss = loss)
    encoder.compile(optimizer = 'rmsprop', loss = loss)
    decoder.compile(optimizer = 'rmsprop', loss = loss)

    vae_2.fit_generator(train_generator, steps_per_epoch = 4000, validation_data = val_generator,
                        epochs=7, validation_steps= 500)

    vae_2.save(os.path.join("model_file_path", "model_file_name"))
    #tf.saved_model.simple_save(K.get_session(), "simple_save_models", {'myInput':input}, {'myOutput':output})

