#############
## Imports ##
#############

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

#################
## Final Model ##
#################

def wsi_segmenter(img_size):
    encoder = get_encoder(img_size)
    decoder = get_decoder(img_size)
    input_encoder = keras.layers.Input(shape=(img_size, img_size, 3))
    encodings = encoder(input_encoder)
    output_decoder = decoder(encodings)
    model = keras.models.Model(input_encoder, output_decoder)
    model.compile(optimizer="adam", loss=binary_focal_loss(), metrics=["acc", f1_m])
    return model

#############
## Encoder ##
#############

def get_encoder_layer(input_tensor, n_filters):
    x = residual_se_block(input_tensor, n_filters)
    x = residual_se_block(x, n_filters)
    x = keras.layers.Conv2D(n_filters, 2, strides=2, padding="same")(x)
    return x

def get_encoder(img_size):
    input_encoder = keras.layers.Input(shape=(img_size, img_size, 3))
    encodings1 = get_encoder_layer(input_encoder, 64)
    encodings2 = get_encoder_layer(encodings1, 128)
    encodings3 = get_encoder_layer(encodings2, 256)
    encoder_model = keras.models.Model(input_encoder, [encodings1, encodings2, encodings3])
    return encoder_model

#############
## Decoder ##
#############

def get_decoder_layer(input_tensor, n_filters, input_encoder=None):
    if input_encoder is not None:
        input_tensor = keras.layers.Concatenate()([input_tensor, input_encoder])
    x = residual_se_block(input_tensor, n_filters)
    x = residual_se_block(x, n_filters)
    x = keras.layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same")(x)
    return x

def get_decoder(img_size):
    input_decoder1 = keras.layers.Input(shape=(img_size // 2, img_size // 2, 64))
    input_decoder2 = keras.layers.Input(shape=(img_size // 4, img_size // 4, 128))
    input_decoder3 = keras.layers.Input(shape=(img_size // 8, img_size // 8, 256))
    decodings1 = get_decoder_layer(input_decoder3, 128)
    decodings2 = get_decoder_layer(decodings1, 64, input_decoder2)
    decodings3 = get_decoder_layer(decodings2, 32, input_decoder1)
    output_decoder = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(decodings3)
    decoder_model = keras.models.Model([input_decoder1, input_decoder2, input_decoder3], output_decoder)
    return decoder_model

###########
## Utils ##
###########

def global_max_pooling(x):
    x = keras.backend.max(x, axis=1, keepdims=True)
    x = keras.backend.max(x, axis=1, keepdims=True)
    return x

def residual_se_block(input_tensor, n_filters, filter_size=3, r=1.):
    input_tensor = keras.layers.Conv2D(n_filters, 1, padding="same")(input_tensor)
    input_tensor = keras.layers.BatchNormalization()(input_tensor)
    input_tensor = keras.layers.Activation("relu")(input_tensor)
    x = keras.layers.Conv2D(n_filters, filter_size, padding="same")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(n_filters, filter_size, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x_se = keras.layers.Lambda(global_max_pooling)(x)
    x_se = keras.layers.Conv2D(int(n_filters / r), 1, padding="same")(x_se)
    x_se = keras.layers.BatchNormalization()(x_se)
    x_se = keras.layers.Activation("relu")(x_se)
    x_se = keras.layers.Conv2D(n_filters, 1, padding="same")(x_se)
    x_se = keras.layers.BatchNormalization()(x_se)
    x_se = keras.layers.Activation("sigmoid")(x_se)
    x = keras.layers.Multiply()([x, x_se])
    output_tensor = keras.layers.Add()([x, input_tensor])
    return output_tensor

######################
## Metrics & Losses ##
######################

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * precision * recall / (precision + recall + K.epsilon())

def binary_focal_loss(gamma=2., alpha=.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed
