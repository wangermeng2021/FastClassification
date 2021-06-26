
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import math
import matplotlib.pyplot as plt
import logging
from utils.lr_scheduler import get_lr_scheduler
from tqdm import tqdm
from utils.common import move_plot_window_to_center
class LRFinder():
    def __init__(self,start_lr:float=1e-7, end_lr:float=1.0,num_it:int=100):
        self.start_lr=start_lr
        self.num_it=num_it
        self.end_lr=end_lr
        self.lrs=[]
        self.best_loss=np.inf
        self.losses=[]
        self.ema_loss=0.0
        self.beta=0.95
        self.batch_index=0

    def find_lr(self,train_generator,model,loss_fn,optimizer,args):
        optimizer.learning_rate.assign(self.start_lr)
        original_model_weights=model.get_weights()
        original_lr = optimizer.learning_rate.numpy()
        self.model=model
        epoch_batch_index=0
        for epoch in range(int(args.epochs)):
            lr = get_lr_scheduler(args)(epoch)
            optimizer.learning_rate.assign(lr)
            train_loss = 0
            train_generator_tqdm = tqdm(enumerate(train_generator), total=len(train_generator))
            for batch_index, (batch_imgs, batch_labels) in train_generator_tqdm:
                with tf.GradientTape() as tape:
                    model_outputs = model(batch_imgs, training=True)
                    data_loss = loss_fn(batch_labels, model_outputs)
                    total_loss = data_loss + args.weight_decay * tf.add_n(
                        [tf.nn.l2_loss(v) for v in model.trainable_variables if '_bn' not in v.name])
                grads = tape.gradient(total_loss, model.trainable_variables)
                if args.optimizer.startswith('SAM'):
                    optimizer.first_step(grads, model.trainable_variables)
                    with tf.GradientTape() as tape:
                        model_outputs = model(batch_imgs, training=True)
                        data_loss = loss_fn(batch_labels, model_outputs)
                        total_loss = data_loss + args.weight_decay * tf.add_n(
                            [tf.nn.l2_loss(v) for v in model.trainable_variables if '_bn' not in v.name])
                    grads = tape.gradient(total_loss, model.trainable_variables)
                    optimizer.second_step(grads, model.trainable_variables)
                else:
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if epoch_batch_index > 0:
                    loss = data_loss.numpy()
                    self.losses.append(loss)

                    if epoch_batch_index > self.num_it // 2 and (math.isnan(loss) or loss > self.best_loss * 5 or epoch_batch_index >= self.num_it - 1):
                        model.set_weights(original_model_weights)
                        optimizer.learning_rate.assign(original_lr)

                        return
                    if epoch_batch_index >= self.num_it // 2 and loss < self.best_loss:
                        self.best_loss = loss
                        self.best_lr = optimizer.learning_rate.numpy()
                lr = self.start_lr * (float(self.end_lr) / float(self.start_lr)) ** (
                            float(epoch_batch_index) / float(self.num_it))
                self.lrs.append(lr)
                optimizer.learning_rate.assign(lr)
                epoch_batch_index+=1
                train_loss += data_loss
                train_generator_tqdm.set_description(
                    "epoch:{}/{},train_loss:{:.4f},lr:{:.6f}".format(epoch + 1, args.epochs,
                                                                     train_loss / (batch_index + 1),
                                                                     optimizer.learning_rate.numpy()))
            train_generator.on_epoch_end()


        model.set_weights(original_model_weights)
        optimizer.learning_rate.assign(original_lr)


    def plot_loss(self, n_skip_beginning=1, n_skip_end=1, x_scale='log'):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale(x_scale)
        move_plot_window_to_center(plt)
        man = plt.get_current_fig_manager()
        man.canvas.set_window_title("lr finder")
        plt.show()

    def plot_loss_change(self, sma=1, n_skip_beginning=1, n_skip_end=1, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        # plt.ylim(y_lim)
        plt.show()

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma=1, n_skip_beginning=10, n_skip_end=5):
        if False:
            derivatives = self.get_derivatives(sma)
            best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
            return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]
        else:
            return self.best_lr*0.5







