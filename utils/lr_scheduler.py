import tensorflow as tf
import numpy as np

def get_lr_scheduler(args):
    warmup_lr=args.init_lr/25
    # args.warmup_epochs=args.epochs//2
    if args.lr_scheduler == 'step':
        def scheduler(epoch,lr=0.001):
            if epoch < args.warmup_epochs:
                current_epoch_lr = warmup_lr + epoch * (args.init_lr -warmup_lr) / args.warmup_epochs
                return current_epoch_lr
            else:
                for index, val in enumerate(args.lr_decay_epoch):
                    if epoch < val:
                        return args.init_lr*args.lr_decay**index
            return args.init_lr*args.lr_decay**len(args.lr_decay_epoch)
        return scheduler
    elif args.lr_scheduler == 'cosine':
        def scheduler(epoch,lr=0.001):

            if epoch < args.warmup_epochs:
                current_epoch_lr = warmup_lr + epoch * (args.init_lr - warmup_lr) / args.warmup_epochs
            else:
                current_epoch_lr = args.init_lr * (
                        1.0 + tf.math.cos(np.pi / (args.epochs - args.warmup_epochs) * (epoch - args.warmup_epochs))) / 2.0
            # print("lr:",current_epoch_lr)
            return current_epoch_lr


        return scheduler
    else:
        raise ValueError("{} is not supported!".format(args.lr_scheduler))

