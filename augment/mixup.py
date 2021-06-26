# # from __future__ import print_function

import numpy as np
class Mixup:
    def __init__(self,beta=1,prob=1.):
        self.beta=beta
        self.prob=prob
    def distort(self,batch_imgs,one_hot_batch_labels):
        if np.random.uniform() < self.prob:
            batch_size = batch_imgs.shape[0]
            random_batch_indexes = np.random.choice(batch_size, batch_size,replace=False)
            mix_ratio_1 = np.random.beta(self.beta, self.beta)
            mix_ratio_2 = 1. - mix_ratio_1
            batch_imgs = mix_ratio_1 * batch_imgs+mix_ratio_2*batch_imgs[random_batch_indexes]

            one_hot_batch_labels = mix_ratio_1 * one_hot_batch_labels + mix_ratio_2 * one_hot_batch_labels[random_batch_indexes]
        return batch_imgs,one_hot_batch_labels

