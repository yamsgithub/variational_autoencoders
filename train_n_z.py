# now load attribute

# 1.A.2

import sys
import os
args = sys.argv

model_file_path = "models"
model_file_name = "vae_model_70_20_10"
log_file_dir = "scalars"
if len(args) >= 2:
    model_file_name = args[1]
if len(args) >= 3:
    log_file_dir = args[2]

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

#fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# for i in range(3):
#     ax[i].imshow(get_input(img_path[i+3]))
#     ax[i].axis('off')
#     ax[i].set_title(img_path[i+3][-10:])
# plt.show()

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
from time import time
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback

class ConvVAE:
    def build(self, input_shape, bottleneck_size, sampling, batch_size = 32):
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
        # DECODER
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

        print('ENCODER ', encoder(input),'\n', len(encoder(input)),'\n', encoder(input)[2])
        output_2 = decoder(encoder(input)[2])
        print('OUTPUT 2', output_2)
        vae = Model(input, output_2, name ='vae')
        return vae, input, output, encoder, decoder, z_mean, z_log_sigma

    # Utility function to convert a tensor into a valid image
    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1
        
        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)
        
        # convert to RGB array
        x *= 255
        x = x.transpose((0, 1, 2))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # Function to visualize the activate of given filter in the given layer
    def visualize_activation(self, layer_name, filter_index, layer_dict, model):
        iterate = self.get_iterate(layer_name, filter_index, layer_dict, model)
    
        input_img_data = np.random.random((1,32, 32, 3)) * 20 + 128.
        step = 1
        
        for _ in range(150):
            loss_value, grads_value = iterate([input_img_data])
            # some filters get stuck to 0, we can skip them
            if loss_value <= K.epsilon():
                return None, None
            # Add the gradients to the input image
            input_img_data += grads_value * step
        img = deprocess_image(input_img_data[0])
        return img, loss_value

    # Define a function that returns a function to compute the loss and gradient of the input wrt to the loss for
    # a given layer_name and filter_index
    def get_iterate(self, layer_name, filter_index, layer_dict, model):
    
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer = layer_dict[layer_name]
        layer_output = layer.output
        #layer_output = layer.get_output_at(-1)
        
        if isinstance(layer, Dense):
            loss = K.mean(layer_output[:, filter_index])
        else:  
            loss = K.mean(layer_output[:, :, :, filter_index])

        # We add an input placeholder with a size the same as the input image
        input_img = model.inputs[0]

        print('\nINPUT IMAGE ', input_img)
        
        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        print('\nGRADS ', grads)

        if not grads is None:
            # normalization trick: we normalize the gradient
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        else:
            grads =0.5
        
        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])
        
        return iterate
    
def main():
    img_sample = get_input(img_path[0])
    print(img_sample.shape)
    conv_vae = ConvVAE()
    vae_2, input, output, encoder, decoder, z_mean, z_log_sigma = conv_vae.build(img_sample.shape, n_size, sampling, batch_size = b_size)
    
    import os
    from tensorflow.keras.models import load_model
    print('MODEL FILE NAME ', os.path.join(model_file_path, model_file_name))
    if os.path.exists(os.path.join(model_file_path, model_file_name)):
        loss = vae_loss(z_mean, z_log_sigma)
        vae_2 = load_model('models/vae_model_70_20_10', custom_objects={'sampling':sampling,'loss':loss})
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
        #file_writer = tf.summary.FileWriter("/home/local/peac004/vae/logs/scalars")
        #file_writer.set_as_default()

        # Get the names of each layer in the model
        layer_dict = dict([(layer.name, layer) for layer in vae_2.layers])
        print('LAYER DICT ', layer_dict.keys())
        
        layer_name = 'decoder'
        output_layer = layer_dict[layer_name]
        
        # Get the number of filters in layer_name
        weights = len(output_layer.get_weights())
        print('Layer ', layer_name, ' has weights: ', weights)

        # with tf.variable_scope('conv2d_transpose_3') as scope_conv:
        #     weights = tf.get_variable('weights')
            
        # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
        x_min = tf.reduce_min(weights)
        x_max = tf.reduce_max(weights)
        weights_0_to_1 = (weights - x_min) / (x_max - x_min)
        weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)
        
        # to tf.image_summary format [batch_size, height, width, channels]
        weights_transposed = tf.transpose (weights_0_to_255_uint8, [3, 0, 1, 2])

        # num_filters = len(output_layer.get_weights()[1])
        # print('Layer ', layer_name, ' has ', num_filters, ' filters')
        
        # # Visualize activations of all the filters in the current layer
        # processed_filters = []
        # for i in range(0, num_filters):
        #     img, loss_value = conv_vae.visualize_activation(layer_name, i, layer_dict, vae_2)
        #     if not img is None: 
        #         print('Processed filter ', i)
        #         processed_filters.append((img, loss_value))
        # print('Number of filters with non-zero loss are ', len(processed_filters))
        
        loss_logging_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs:  tf.compat.v1.summary.scalar('loss', tensor=logs['loss']),
            on_batch_end=lambda batch, logs:  tf.compat.v1.summary.scalar('loss', tensor=logs['loss'])
        )

        image_logging_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: tf.image_summary('vae_output', weights_transposed, max_images=3)
        )

        tb_batch_callback = TensorBoard(os.path.join('/home/local/peac004/vae/logs',log_file_dir,'{}'.format(time())), write_graph=False, update_freq='batch', profile_batch=0)
        tb_epoch_callback = TensorBoard(os.path.join('/home/local/peac004/vae/logs',log_file_dir,'{}'.format(time())), write_graph=False, write_images=True, update_freq='epoch', profile_batch=0)
        vae_2.fit_generator(train_generator, steps_per_epoch = 50, validation_data = val_generator,
                            epochs=7, validation_steps= 500, callbacks=[tb_batch_callback, tb_epoch_callback, loss_logging_callback, image_logging_callback])
        
        vae_2.save(os.path.join(model_file_path, model_file_name))
        #tf.saved_model.simple_save(K.get_session(), "simple_save_models", {'myInput':input}, {'myOutput':output})

if __name__=='__main__':
    main()
