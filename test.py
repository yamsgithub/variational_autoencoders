import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from utils  import sampling, vae_loss
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from data_generator import get_input
import numpy as np

#vae_2 = load_model('models/vae_model_70_20_10', custom_objects={'sampling':sampling,'loss':vae_loss})

import pandas as pd
attr = pd.read_csv('data/list_attr_celeba.csv')
attr = attr.set_index('image_id')

IMG_NAME_LENGTH = 6
file_path = "data/img_align_celeba/"
img_id = np.arange(1,len(attr.index)+1)
img_path = []
for i in range(len(img_id)):
    img_path.append(file_path + (IMG_NAME_LENGTH - len(str(img_id[i])))*'0' + str(img_id[i]) + '.jpg')

with tf.Session(graph=tf.Graph()) as sess:
    vae_2 = tf.saved_model.load(sess, ["serve"], 'models')
        
    inputImage_name = vae_2.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['myInput'].name
    inputImage = tf.get_default_graph().get_tensor_by_name(inputImage_name)

    outputPrediction_name = vae_2.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['myOutput'].name
    outputPrediction = tf.get_default_graph().get_tensor_by_name(outputPrediction_name)
      
    import random
    x_test = []
    for i in range(64):
        x_test.append(get_input(img_path[random.randint(0,len(img_id))]))
    x_test = np.array(x_test)
    #figure_Decoded = vae_2.predict(x_test.astype('float32')/127.5 -1, batch_size = b_size)
    figure_Decoded = sess.run(outputPrediction, feed_dict={inputImage:x_test})
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
