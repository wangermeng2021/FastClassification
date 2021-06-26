
import tensorflow as tf
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW
from optimizer.sam import SAMOptimizer
def get_optimizer(args):

    if args.optimizer == "Adam":
        return tf.keras.optimizers.Adam()
    elif args.optimizer == "SGD":
        return tf.keras.optimizers.SGD()
    elif args.optimizer == "AdamW":
        return AdamW(weight_decay=args.weight_decay)
    elif args.optimizer == "SAM-SGD":
        return SAMOptimizer(tf.keras.optimizers.SGD())
    elif args.optimizer == "SAM-Adam":
        return SAMOptimizer(tf.keras.optimizers.Adam())
    else:
        raise ValueError("{} is not supported!".format(args.optimizer))
