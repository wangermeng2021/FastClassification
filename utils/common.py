import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import backend as K
import glob
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from utils.confusion_matrix_pretty_print import plot_confusion_matrix_from_data
from screeninfo import get_monitors

def set_mixed_precision(policy_name='mixed_float16'):

    tf_version=float('.'.join(tf.__version__.split('.')[:-1]))
    if tf_version>=2.4:
        tf.keras.mixed_precision.set_global_policy(policy_name)
    else:
        policy = tf.keras.mixed_precision.experimental.Policy(policy_name)
        tf.keras.mixed_precision.experimental.set_policy(policy)


def move_plot_window_to_center(plt):
    fig = plt.gcf()
    size = fig.get_size_inches() * fig.dpi  # size in pixels
    mngr = plt.get_current_fig_manager()
    try:
        mngr.window.setGeometry(get_monitors()[0].width//2-size[0]//2, get_monitors()[0].height//2-size[1]//2, size[0], size[1])
    except:
        mngr.window.setGeometry=(get_monitors()[0].width//2-size[0]//2, get_monitors()[0].height//2-size[1]//2, size[0], size[1])

def clean_checkpoints_fn(checkpoints_dir):
    max_keep_num = 5
    while True:
        checkpoints_list = glob.glob(os.path.join(checkpoints_dir, "*.index"))
        if len(checkpoints_list) > max_keep_num:
            sorted_by_mtime_ascending = sorted(checkpoints_list, key=lambda t: os.stat(t).st_mtime)
            for i in range(len(checkpoints_list) - max_keep_num):
                try:
                    os.remove(sorted_by_mtime_ascending[i])
                    os.remove(".".join(sorted_by_mtime_ascending[i].split('.')[:-1]) + ".data-00000-of-00001")
                except:
                    pass
        time.sleep(5)
def clean_checkpoints(checkpoints_dir):
    x = threading.Thread(target=clean_checkpoints_fn, args=(checkpoints_dir,))
    x.start()

def show_classes_hist(class_sizes,class_names):
    plt.rcdefaults()
    x_pos = np.arange(len(class_names))
    plt.bar(x=x_pos,height=class_sizes,width=0.4,align="center")
    # plt.ylabel(u'count')
    plt.xlabel(u'class name')
    plt.xticks(x_pos,class_names)
    # plt.legend((rect,),(u"xxx",))
    move_plot_window_to_center(plt)
    man = plt.get_current_fig_manager()
    man.canvas.set_window_title("classes histogram")
    plt.show()

def get_best_model_path(dir):
    files = glob.glob(os.path.join(dir,"best_weight*"))
    sorted_by_mtime_descending = sorted(files, key=lambda t: -os.stat(t).st_mtime)
    if len(sorted_by_mtime_descending)==0:
        return None
    return '.'.join(sorted_by_mtime_descending[0].split('.')[0:-1])


def freeze_model(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_model(l, frozen)


def plot_figures(figures, nrows = 1, ncols=1,window_title="training samples"):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,dpi=180)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    move_plot_window_to_center(plt)
    man = plt.get_current_fig_manager()
    man.canvas.set_window_title(window_title)
    plt.show()



def show_training_images(generator,num_img=9):
    prefix="img_"
    figures={}
    img_index=0
    for index,(image,label) in enumerate(generator):
        title=prefix+str(index)+":"+generator.class_names[np.argmax(label[0])]
        figures[title]=image[0].astype(np.uint8)
        img_index+=1
        if img_index>=num_img:
            break
    # plot of the images
    num_row=len(list(figures.keys()))//3
    num_col=3
    plot_figures(figures, num_row, num_col,window_title="training samples")


def get_confusion_matrix(dataset,model,save_dir=None):
    pred_result = model.predict(dataset,verbose=1)
    pred_cls = np.argmax(pred_result,axis=-1)
    pred_scores = np.max(pred_result,axis=-1)
    valid_img_path_list = dataset.img_path_list[dataset.valid_mask]
    valid_label_list = dataset.label_list[dataset.valid_mask]
    wrong_pred_mask = valid_label_list!=pred_cls

    columns = dataset.class_names
    annot = True;
    cmap = 'Oranges';
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    #size::
    fz = 12;
    figsize = [9,9];
    if(len(valid_label_list) > 10):
        fz=9; figsize=[14,14];

    cur_plt=plot_confusion_matrix_from_data(valid_label_list, pred_cls, columns,
    annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)
    move_plot_window_to_center(plt)
    man = plt.get_current_fig_manager()
    man.canvas.set_window_title("confusion_matrix")
    plt.show()

    #
    return show_multi_img(dataset,valid_img_path_list[wrong_pred_mask],valid_label_list[wrong_pred_mask],pred_cls[wrong_pred_mask],pred_scores[wrong_pred_mask],dataset.class_names)

def show_multi_img(dataset,img_path_list,label_list,pred_list,pred_scores,class_names):
    prefix=""
    # max_num = min(len(img_path_list),10)
    # img_path_list=img_path_list[:max_num]
    # label_list=label_list[:max_num]
    # pred_list=pred_list[:max_num]
    for index,(img_path,label,pred,score) in enumerate(zip(img_path_list,label_list,pred_list,pred_scores)):
        title=prefix+"{}/{} < ".format(str(index+1),len(img_path_list))+"label:"+class_names[label]+",pred:"+class_names[pred]+",score:"+"{:.2f}".format(score)+" >"+"\n"+img_path
        image = np.ascontiguousarray(Image.open(img_path).convert('RGB'))
        image = dataset.valid_resize_img(image)
        plt.imshow(image)
        plt.title(title)
        move_plot_window_to_center(plt)
        man = plt.get_current_fig_manager()
        man.canvas.set_window_title("wrong prediction images")
        plt.show()
        plt.close()
    return list(zip(img_path_list,label_list,pred_list,pred_scores))