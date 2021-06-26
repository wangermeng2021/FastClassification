
import tensorflow as tf
# from losses.focal import get_focal
def get_losses(args):

    if args.loss == "ce":
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    elif args.loss == "bce":
        return tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
    # elif args.loss == "focal":
    #     return get_focal(args)
    else:
        raise ValueError("{} is not supported!".format(args.model))
