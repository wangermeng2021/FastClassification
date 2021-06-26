
import tensorflow as tf
from utils.dropblock import dropblock
def rename_layer(name="last_conv"):
    def layer_func(x):
        return x
    return tf.keras.layers.Lambda(layer_func, name=name)


class dropblock_layer(tf.keras.layers.Layer):
    def __init__(self,):
        super(dropblock_layer, self).__init__()
    def call(self, inputs, training=None):
        # tf.print("training: ", training)
        if training:
            inputs = dropblock(inputs, keep_prob=0.9, dropblock_size=3)
        return inputs


# def dropblock_layer():
#     # def layer_func(x):
#     #     x = dropblock(x, keep_prob=0.9, dropblock_size=7)
#     #     return x
#     return tf.keras.layers.Lambda(lambda x: dropblock(x, keep_prob=0.9, dropblock_size=7))
