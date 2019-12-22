import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape = (z_mean.get_shape().as_list()[1],) , mean = 0, stddev = 1)
    #print('EPSILON ', epsilon,'\n',epsilon.shape,' \n MEAN', K.get_value(K.mean(epsilon)))
    print('\n\n\nZ MEAN ', Flatten()(z_mean))
    print('Z LOG SIGMA ', Flatten()(z_log_sigma),'\n\n\n')
    return z_mean + K.exp(z_log_sigma/2) * epsilon

def vae_loss(z_mean, z_log_sigma):
    def loss(input_img, output):
        from tensorflow.keras.losses import mse
        # Compute error in reconstruction
        reconstruction_loss = mse(K.flatten(input_img) , K.flatten(output))

        # Compute the KL Divergence regularization term
        kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = -1)
        # Return the average loss over all images in batch
        total_loss = (reconstruction_loss + 0.0001 * kl_loss)
        return total_loss
    return loss


