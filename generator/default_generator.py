import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
from tqdm import tqdm
import albumentations as A
import random
from utils.custom_augment import train_augment
from utils.preprocess import normalize
from augment import auto_augment,rand_augment,baseline_augment,mixup,cutmix
from utils.preprocess import resize_img_aug,resize_img
class DefaultGenerator(tf.keras.utils.Sequence):
    def __init__(self, args, mode='train',train_valid_split_ratio=0.7,dataset_sample_ratio=1.0):
        self.args = args
        self.dataset_dir = args.dataset_dir
        batch_size = args.batch_size
        augment = args.augment

        self.class_counts=[]
        self.train_valid_split_ratio=train_valid_split_ratio
        random.seed(123)
        # np.random.seed(123)
        self.set_img_size(args.progressive_resizing[-1])

        (train_img_path_list,val_img_path_list,train_label_list,val_label_list,class_names)=self.create_split_list(self.dataset_dir,dataset_sample_ratio)
        # print(train_img_path_list)
        # print(val_img_path_list)
        if mode=='train':
            self.img_path_list=train_img_path_list
            self.label_list = train_label_list
            img_path_list_len = len(self.img_path_list)
            pad_len = batch_size-img_path_list_len%batch_size
            pad_img_path_list=[]
            pad_label_list=[]
            for _ in range(pad_len):
                rand_index = np.random.randint(0,img_path_list_len)
                pad_img_path_list.append(self.img_path_list[rand_index])
                pad_label_list.append(self.label_list[rand_index])

            self.img_path_list = np.append(self.img_path_list, pad_img_path_list)
            self.label_list = np.append(self.label_list, pad_label_list)
            self.data_index = np.arange(0, len(self.label_list))
            np.random.shuffle(self.data_index)

        else:
            self.img_path_list=val_img_path_list
            self.label_list = val_label_list
            self.data_index = np.arange(0, len(self.label_list))
            self.valid_mask=[True]*len(val_label_list)
        self.img_path_list = np.array(self.img_path_list)
        self.label_list = np.array(self.label_list)
        self.augment = augment
        self.mode = mode
        self.batch_size = batch_size
        self.eppch_index = 0
        self.class_names=class_names
        self.num_class = len(class_names)


        self.auto_augment = auto_augment.AutoAugment(augmentation_name='v0')
        self.rand_augment = rand_augment.RandAugment(num_layers=2,magnitude=10.)
        self.mixup = mixup.Mixup(beta=1,prob=1.)
        self.cutmix = cutmix.Cutmix(beta=1,prob=1.)
        self.baseline_augment = baseline_augment.Baseline()



    def read_img(self, path):
        image = np.ascontiguousarray(Image.open(path).convert('RGB'))
        return image
        # return image[:, :, ::-1]
    def on_epoch_end(self):
        if self.mode == 'train':
            np.random.shuffle(self.data_index)
        # else:
        #     self.valid_mask=[True]*len(self.label_list)
        self.eppch_index += 1
    def __len__(self):
        return int(np.ceil(len(self.img_path_list) / self.batch_size))
    def __getitem__(self, batch_index):

        batch_img_paths = self.img_path_list[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        batch_labels = self.label_list[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        one_hot_batch_labels = np.zeros([len(batch_img_paths), self.num_class])
        batch_imgs = []
        if self.mode == "valid":
            valid_index = 0
            for i in range(len(batch_img_paths)):
                try:
                    img = self.read_img(batch_img_paths[i])
                except:
                    self.valid_mask[batch_index * self.batch_size+i]=False
                    continue

                img = self.valid_resize_img(img)
                if self.args.weights:
                    if self.args.backbone[0:3] == "Res":
                        img = normalize(img, mode='caffe')
                else:
                    img = normalize(img, mode='tf')

                batch_imgs.append(img)
                one_hot_batch_labels[valid_index, batch_labels[i]] = 1
                valid_index += 1
            batch_imgs = np.array(batch_imgs)
            one_hot_batch_labels=one_hot_batch_labels[:valid_index]
        else:

            valid_index=0
            for i in range(len(batch_img_paths)):
                try:
                    img = self.read_img(batch_img_paths[i]).astype(np.uint8)
                except:
                    continue

                img = self.train_resize_img(img)

                if self.augment == 'auto_augment':
                    img = self.auto_augment.distort(tf.constant(img)).numpy()
                elif self.augment == 'rand_augment':
                    img = self.rand_augment.distort(tf.constant(img)).numpy()
                elif self.augment == 'baseline':
                    img = self.baseline_augment.distort(img)
                if self.args.weights:
                    if self.args.backbone[0:3] == "Res":
                        img = normalize(img,mode='caffe')
                else:
                    img = normalize(img, mode='tf')
                batch_imgs.append(img)
                one_hot_batch_labels[valid_index, batch_labels[i]] = 1
                valid_index += 1
            batch_imgs = np.array(batch_imgs)
            one_hot_batch_labels = one_hot_batch_labels[:valid_index]
            # # if np.random.rand(1) < 0.5:
            if self.augment == 'cutmix':
                batch_imgs, one_hot_batch_labels = self.cutmix.distort(batch_imgs,one_hot_batch_labels)
            elif self.augment == 'mixup':
                batch_imgs, one_hot_batch_labels = self.mixup.distort(batch_imgs,one_hot_batch_labels)

        return batch_imgs, one_hot_batch_labels

    def create_split_list(self,root_dir,dataset_sample_ratio):
        train_imgs_list = []
        val_imgs_list = []
        train_labels_list = []
        val_labels_list = []
        valid_class_names=[]
        class_names = os.listdir(root_dir)
        index=0
        for class_name in class_names:
            sub_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(sub_dir):
                valid_class_names.append(class_name)
                file_list = []
                file_name_list = os.listdir(sub_dir)
                for file_name in tqdm(file_name_list):

                    img_path=os.path.join(sub_dir, file_name)

                    if os.path.isdir(img_path) or file_name.split('.')[-1].lower() not in ['jpg','jpeg','png','bmp']:
                        continue

                    # try:
                    #     tmp_img = cv2.imread(os.path.join(sub_dir, file_name))
                    # except:
                    #     continue
                    file_list.append(os.path.join(sub_dir, file_name))

                num_file = int(len(file_list)*dataset_sample_ratio)
                file_list = random.sample(file_list, num_file)
                num_train = int(num_file * self.train_valid_split_ratio)
                train_imgs_list.extend(file_list[:num_train])
                val_imgs_list.extend(file_list[num_train:])
                train_labels_list.extend([index] * num_train)
                val_labels_list.extend([index] * (num_file - num_train))
                index+=1
                self.class_counts.append(num_file)


        train_imgs_list = np.array(train_imgs_list)
        train_labels_list = np.array(train_labels_list)
        val_imgs_list = np.array(val_imgs_list)
        val_labels_list = np.array(val_labels_list)

        random_index = random.sample(range(len(train_imgs_list)), len(train_imgs_list))
        train_imgs_list = train_imgs_list[random_index]
        train_labels_list = train_labels_list[random_index]
        random_index = random.sample(range(len(val_imgs_list)), len(val_imgs_list))
        val_imgs_list = val_imgs_list[random_index]
        val_labels_list = val_labels_list[random_index]

        with open(os.path.join(root_dir,'class.names'),'w') as f1:
            for val in valid_class_names:
                f1.write(val + "\n")
        return train_imgs_list,val_imgs_list,train_labels_list,val_labels_list,valid_class_names

    def valid_resize_img(self, img):
        dst_size=self.resize_size
        img,_,_  = resize_img(img, dst_size)
        img = self.center_crop_transform(image=img)['image']
        return img
    def train_resize_img(self,img):
        dst_size=self.resize_size
        img,_,_  = resize_img_aug(img,dst_size)
        img= self.random_crop_transform(image=img)['image']
        return img

    def set_img_size(self,img_size):
        img_w=img_size[0]
        img_h=img_size[1]
        self.resize_size = ( int(img_w/0.875),  int(img_h/0.875))
        self.crop_size = (img_w, img_h)
        self.random_crop_transform = A.Compose([
            A.RandomCrop(width=self.crop_size[0], height=self.crop_size[1]),
        ])
        self.center_crop_transform = A.Compose([
            A.CenterCrop(width=self.crop_size[0], height=self.crop_size[1]),
        ])
