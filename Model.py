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
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc", f1_metric])
    return model

#############
## Encoder ##
#############

def get_encoder_layer(input_tensor, n_filters):
    x = residual_conv_block(input_tensor, n_filters)
    x = residual_conv_block(x, n_filters)
    x = keras.layers.Conv2D(n_filters, 3, strides=2, padding="same")(x)
    return x

def get_encoder(img_size):
    input_encoder = keras.layers.Input(shape=(img_size, img_size, 3))
    encodings1 = get_encoder_layer(input_encoder, 64)
    encodings2 = get_encoder_layer(encodings1, 128)
    encodings3 = get_encoder_layer(encodings2, 256)
    encodings4 = get_encoder_layer(encodings3, 512)
    encodings5 = get_encoder_layer(encodings4, 1024)
    encoder_model = keras.models.Model(input_encoder, [encodings1, encodings2, encodings3, encodings4, encodings5])
    return encoder_model

#############
## Decoder ##
#############

def get_decoder_layer(input_tensor, n_filters, input_encoder=None):
    if input_encoder is not None:
        input_tensor = keras.layers.Concatenate()([input_tensor, input_encoder])
    x = residual_unconv_block(input_tensor, n_filters)
    x = residual_unconv_block(x, n_filters)
    x = keras.layers.Conv2DTranspose(n_filters, 3, strides=2, padding="same")(x)
    return x

def get_decoder(img_size):
    input_decoder1 = keras.layers.Input(shape=(img_size // 2, img_size // 2, 64))
    input_decoder2 = keras.layers.Input(shape=(img_size // 4, img_size // 4, 128))
    input_decoder3 = keras.layers.Input(shape=(img_size // 8, img_size // 8, 256))
    input_decoder4 = keras.layers.Input(shape=(img_size // 16, img_size // 16, 512))
    input_decoder5 = keras.layers.Input(shape=(img_size // 32, img_size // 32, 1024))
    decodings1 = get_decoder_layer(input_decoder5, 512)
    decodings2 = get_decoder_layer(decodings1, 256, input_decoder4)
    decodings3 = get_decoder_layer(decodings2, 128, input_decoder3)
    decodings4 = get_decoder_layer(decodings3, 64, input_decoder2)
    decodings5 = get_decoder_layer(decodings4, 64, input_decoder1)
    output_decoder = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(decodings5)
    decoder_model = keras.models.Model([input_decoder1, input_decoder2, input_decoder3, input_decoder4, input_decoder5], output_decoder)
    return decoder_model

###########
## Utils ##
###########

def residual_conv_block(input_tensor, n_filters, filter_size=3, r=1.):
    input_tensor = keras.layers.Conv2D(n_filters, 1, padding="same")(input_tensor)
    input_tensor = keras.layers.BatchNormalization()(input_tensor)
    input_tensor = keras.layers.Activation("relu")(input_tensor)
    x = keras.layers.Conv2D(n_filters, filter_size, padding="same")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(n_filters, filter_size, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    output_tensor = keras.layers.Add()([x, input_tensor])
    return output_tensor

def residual_unconv_block(input_tensor, n_filters, filter_size=3, r=1.):
    input_tensor = keras.layers.Conv2DTranspose(n_filters, 1, padding="same")(input_tensor)
    input_tensor = keras.layers.BatchNormalization()(input_tensor)
    input_tensor = keras.layers.Activation("relu")(input_tensor)
    x = keras.layers.Conv2DTranspose(n_filters, filter_size, padding="same")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2DTranspose(n_filters, filter_size, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    output_tensor = keras.layers.Add()([x, input_tensor])
    return output_tensor

######################
## Metrics & Losses ##
######################

def f1_metric(y_true, y_pred):

    def precision_metric(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall_metric(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * precision * recall / (precision + recall + K.epsilon())

# def dice_loss(y_true, y_pred):
#     numerator = 2 * tf.reduce_mean(y_true * y_pred, axis=-1)
#     denominator = tf.reduce_mean(y_true + y_pred, axis=-1)
#     return 1 - (numerator + 1) / (denominator + 1)

# def total_loss(y_true, y_pred):
#     bce = keras.losses.binary_crossentropy(y_true, y_pred)
#     dice = dice_loss(y_true, y_pred)
#     return 0.8 * bce + 0.2 * dice
