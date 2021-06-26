
import tensorflow as tf
from utils.layers import rename_layer
import cv2
from utils.class_head import class_head
class ResNet():
    def __init__(self, classes=10, type=50, concat_max_and_average_pool=False,weights='imagenet',loss='ce'):
        self.type = type
        self.classes = classes
        self.concat_max_and_average_pool = concat_max_and_average_pool
        self.weights = weights
        self.loss = loss
    def get_model(self):
        input_layer = tf.keras.Input(shape=(None, None, 3))
        if self.type == 50:
            try:
                x = tf.keras.applications.ResNet50(include_top=False,weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.resnet50.ResNet50(include_top=False,weights=self.weights)(input_layer)
        elif self.type == 101:
            try:
                x = tf.keras.applications.ResNet101(include_top=False,weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.resnet.ResNet101(include_top=False,weights=self.weights)(input_layer)
        elif self.type == 152:
            try:
                x = tf.keras.applications.ResNet152(include_top=False,weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.resnet.ResNet152(include_top=False,weights=self.weights)(input_layer)
        else:
            raise ValueError('Unsupported ResNet type:{}'.format(self.type))

        if self.concat_max_and_average_pool:
            x1 = tf.keras.layers.GlobalAveragePooling2D()(x)
            x2 = tf.keras.layers.GlobalMaxPooling2D()(x)
            x = rename_layer(name="last_conv")(x)
            x = tf.keras.layers.Concatenate()([x1,x2])
        else:
            x = rename_layer(name="last_conv")(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = class_head(x, self.classes, 512)
        if self.loss == 'ce':
            x = tf.keras.layers.Activation(tf.keras.activations.softmax, name="predictions")(x)
        else:
            x = tf.keras.layers.Activation(tf.keras.activations.sigmoid, name="predictions")(x)

        return tf.keras.Model(inputs=input_layer, outputs=x)

