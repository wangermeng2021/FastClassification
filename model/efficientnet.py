
import tensorflow as tf
from utils.layers import rename_layer
from utils.class_head import class_head
from utils.layers import dropblock_layer


# class MyLayer(tf.keras.layers.Layer):
#     @tf.function
#     def call(self, inputs, training=None):
#         tf.print("training: ", training)
#         tf.print("K.learning_phase(): ", K.learning_phase())
#         if training:
#             x = dropblock(x, keep_prob=0.9, dropblock_size=7, data_format='channels_last')
#         return keras.backend.in_test_phase(inputs + 1., inputs + 2., training)



class EfficientNet():
    def __init__(self, classes=10, type='B0', concat_max_and_average_pool=False,weights='imagenet',loss='ce',dropout=0.1):
        self.type = type
        self.classes = classes
        self.concat_max_and_average_pool = concat_max_and_average_pool
        self.efficientnet_input_size = {'B0':224, 'B1':240, 'B2':260, 'B3':300, 'B4':380, 'B5':456,'B6':528, 'B7':600}
        self.weights = weights
        self.loss=loss
        self.dropout=dropout
    def get_model(self):
        input_layer = tf.keras.Input(shape=(None, None, 3))
        # input_layer = tf.keras.Input(shape=(224, 224, 3))

        if self.type == 'B0':
            try:
                x = tf.keras.applications.EfficientNetB0(include_top=False, weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights=self.weights)(input_layer)
        elif self.type == 'B1':
            try:
                x = tf.keras.applications.EfficientNetB1(include_top=False, weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, weights=self.weights)(input_layer)
        elif self.type == 'B2':
            try:
                x = tf.keras.applications.EfficientNetB2(include_top=False, weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, weights=self.weights)(input_layer)
        elif self.type == 'B3':
            try:
                x = tf.keras.applications.EfficientNetB3(include_top=False, weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights=self.weights)(input_layer)
        elif self.type == 'B4':
            try:
                x = tf.keras.applications.EfficientNetB4(include_top=False, weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, weights=self.weights)(input_layer)
        elif self.type == 'B5':
            try:
                x = tf.keras.applications.EfficientNetB5(include_top=False, weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.efficientnet.EfficientNetB5(include_top=False, weights=self.weights)(input_layer)
        elif self.type == 'B6':
            try:
                x = tf.keras.applications.EfficientNetB6(include_top=False, weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.efficientnet.EfficientNetB6(include_top=False, weights=self.weights)(input_layer)
        elif self.type == 'B7':
            try:
                x = tf.keras.applications.EfficientNetB7(include_top=False, weights=self.weights)(input_layer)
            except:
                x = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights=self.weights)(input_layer)

        # x=dropblock_layer()(x)

        if self.concat_max_and_average_pool:
            x1 = tf.keras.layers.GlobalAveragePooling2D()(x)
            x2 = tf.keras.layers.GlobalMaxPooling2D()(x)
            x = rename_layer(name="last_conv")(x)
            x = tf.keras.layers.Concatenate()([x1,x2])
        else:
            x = rename_layer(name="last_conv")(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = class_head(x, self.classes, 512,dropout=self.dropout)

        # print('x.dtype: %s' % x.dtype.name)
        # print(x.kernel)


        if self.loss=='ce':
            x = tf.keras.layers.Activation(tf.keras.activations.softmax,dtype='float32', name="predictions")(x)
        else:
            x = tf.keras.layers.Activation(tf.keras.activations.sigmoid,dtype='float32', name="predictions")(x)
        # print('Outputs dtype: %s' % x.dtype.name)
        return tf.keras.Model(inputs=input_layer, outputs=x)



