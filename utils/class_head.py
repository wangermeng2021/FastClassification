# # from __future__ import print_function
import numpy as np
import cv2
import tensorflow as tf
# def class_head(x_in,num_class,hidden_layer_out=512,dropout=0.1):
#     x=inputs=tf.keras.layers.Input(x_in.shape[1:])
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(dropout)(x)
#     x = tf.keras.layers.Dense(hidden_layer_out, use_bias=False, activation='relu')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(dropout)(x)
#     x = tf.keras.layers.Dense(num_class, use_bias=False, activation=None)(x)
#     return tf.keras.Model(inputs,x,name="class_head")(x_in)
def class_head(x_in,num_class,hidden_layer_out=512,dropout=0.1):
    x=inputs=tf.keras.layers.Input(x_in.shape[1:])
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(num_class, use_bias=False, activation=None)(x)
    return tf.keras.Model(inputs,x,name="class_head")(x_in)
