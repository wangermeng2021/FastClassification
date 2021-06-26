import os,argparse,sys
from utils.lr_scheduler import get_lr_scheduler
from model.model_builder import get_model
from losses.losses_builder import get_losses
from optimizer.optimizer_builder import get_optimizer
from generator.generator_builder import get_generator
from utils.common import freeze_model,show_training_images,get_best_model_path,get_confusion_matrix,show_classes_hist,clean_checkpoints
from utils.lr_finder import LRFinder
import logging
import tensorflow as tf
from utils.common import set_mixed_precision
from tensorboard import program
import webbrowser

# logging.getLogger().setLevel(logging.INFO)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for classification.')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    # parser.add_argument('--img-size', nargs='+', default=[224, 224],type=int)
    parser.add_argument('--progressive-resizing', nargs='+', default=[224,224],type=int)
    parser.add_argument('--dataset-dir', default='/home/wangem1/dataset/imagenette2-320/val', type=str, help="root directory of dataset")
    parser.add_argument('--augment', default='baseline', help="choices=['baseline','mixup','cutmix',rand_augment,auto_augment]")
    parser.add_argument('--dataset-type', default='default', type=str, help="choices=['default']")
    parser.add_argument('--backbone', default='EfficientNetB0', type=str,help="choices=['ResNet50','ResNet101','ResNet152','EfficientNetB0',...]")
    parser.add_argument('--weights', default='imagenet',help="choices=[None,'imagenet','efficientnetb0_notop.h5']")
    parser.add_argument('--concat-max-and-average-pool', default=True, type=bool,help="Use concat_max_and_average_pool layer in model")
    parser.add_argument('--init-lr', default=1e-3,type=float)
    parser.add_argument('--lr-scheduler', default='cosine', type=str,help="choices=['step','cosine']")
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--lr-decay-epoch', default=[80, 150, 180], type=int)
    parser.add_argument('--warmup-epochs', default=0, type=int)
    parser.add_argument('--weight-decay', default=5e-4, type=float)#5e-4
    parser.add_argument('--optimizer', default='SAM-SGD', help="choices=['Adam','SGD','AdamW','SAM-SGD','SAM-Adam']")
    parser.add_argument('--loss', default='ce', help="choices=['ce','bce']")

    parser.add_argument('--train-valid-split-ratio', default=0.7, type=float)
    parser.add_argument('--dataset-sample-ratio', default=1.0,type=float) #for accelerating debugging process
    parser.add_argument('--mixed-precision', default=False,type=bool)
    parser.add_argument('--label-smoothing', default=0.0,type=float)
    parser.add_argument('--dropout', default=0.1,type=float)
    parser.add_argument('--start-eval-epoch', default=1, type=int)
    parser.add_argument('--accumulated-gradient-num', default=1, type=int)
    parser.add_argument('--checkpoints', default='./checkpoints')
    args=parser.parse_args(args)

    assert len(args.progressive_resizing)%2==0, "Invalid progressive resizing, should be divided by 2:{}".format(args.progressive_resizing)
    img_size_list=[]
    for i in range(len(args.progressive_resizing)//2):
        img_size_list.append((args.progressive_resizing[i*2],args.progressive_resizing[i * 2+1]))
    args.progressive_resizing=img_size_list
    return args

import time
import numpy as np
from tqdm import tqdm
def main(args):
    args=parse_args(args)

    #set mixed_precision to float16
    if args.mixed_precision:
        set_mixed_precision('mixed_float16')
    #build dataset and model
    train_generator, val_generator = get_generator(args)
    model = get_model(args,train_generator.num_class)
    loss_fn=get_losses(args)
    optimizer=get_optimizer(args)
    # perform data sanity check to identify invalid inputs
    show_training_images(train_generator,num_img=9)
    # check class imbalance
    show_classes_hist(train_generator.class_counts,train_generator.class_names)
    #clean old checkpoints
    clean_checkpoints(args.checkpoints)
    # #tensorboard
    open_tensorboard_url = False
    os.system('rm -rf ./logs/')
    tb = program.TensorBoard()
    try:
        tb.configure(argv=[None, '--logdir', 'logs','--reload_interval','15','--load_fast','false'])
    except:
        tb.configure(argv=[None, '--logdir', 'logs','--reload_interval','15'])
    url = tb.launch()
    print("Tensorboard engine is running at {}".format(url))

    # create directory to save checkpoints
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    print("loading dataset...")
    start_time = time.perf_counter()
    best_val_loss = np.inf
    best_val_epoch = -1
    train_writer = tf.summary.create_file_writer("logs/train_loss")
    val_writer = tf.summary.create_file_writer("logs/val_loss")
    acc_writer = tf.summary.create_file_writer("logs/acc")

    # lr finder
    if args.init_lr == 0:
        print("\nlr finder is running...")
        lr_finder = LRFinder(start_lr=1e-7, end_lr=1.0, num_it=max(min(len(train_generator) // 2,100),30))
        lr_finder.find_lr(train_generator,model,loss_fn,optimizer,args)
        # show training loss
        lr_finder.plot_loss()
        # lr_finder.plot_loss_change()
        best_init_lr = lr_finder.get_best_lr()
        print("\nbest_init_lr:{}".format(best_init_lr))
        args.init_lr = best_init_lr

    #training
    for epoch in range(int(args.epochs)):
        lr = get_lr_scheduler(args)(epoch)
        optimizer.learning_rate.assign(lr)
        remaining_epoches = args.epochs - epoch - 1
        epoch_start_time = time.perf_counter()

        if args.progressive_resizing:
            img_size_index = int(epoch // np.ceil(args.epochs / len(args.progressive_resizing)))
            img_size=args.progressive_resizing[img_size_index]
            train_generator.set_img_size(img_size)
            print("progressive resizing:",img_size)
        train_loss = 0
        train_generator_tqdm = tqdm(enumerate(train_generator), total=len(train_generator))
        for batch_index, (batch_imgs, batch_labels)  in train_generator_tqdm:
            with tf.GradientTape() as tape:
                model_outputs = model(batch_imgs, training=True)
                data_loss = loss_fn(batch_labels, model_outputs)
                total_loss = data_loss + args.weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if '_bn' not in v.name])
            grads = tape.gradient(total_loss, model.trainable_variables)
            if args.optimizer.startswith('SAM'):
                optimizer.first_step(grads, model.trainable_variables)
                with tf.GradientTape() as tape:
                    model_outputs = model(batch_imgs, training=True)
                    data_loss = loss_fn(batch_labels, model_outputs)
                    total_loss = data_loss + args.weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if '_bn' not in v.name])
                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.second_step(grads, model.trainable_variables)
            else:
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss += data_loss
            train_generator_tqdm.set_description(
                "epoch:{}/{},train_loss:{:.4f},lr:{:.6f}".format(epoch+1, args.epochs,
                                                                 train_loss/(batch_index+1),
                                                                 optimizer.learning_rate.numpy()))
        train_generator.on_epoch_end()

        with train_writer.as_default():
            tf.summary.scalar("train_loss-val_loss", train_loss/(len(train_generator)*args.batch_size), step=epoch)
            train_writer.flush()
        #evaluation
        if epoch >= args.start_eval_epoch-1:
            val_loss = 0
            val_acc = 0
            num_img = 0
            cur_val_loss=cur_val_acc=0
            val_generator_tqdm = tqdm(enumerate(val_generator), total=len(val_generator))
            for batch_index, (batch_imgs, batch_labels) in val_generator_tqdm:
                model_outputs = model(batch_imgs)
                total_loss = loss_fn(batch_labels, model_outputs)
                val_loss += total_loss
                wrong_pred_mask = np.argmax(batch_labels, axis=-1) == np.argmax(model_outputs, axis=-1)
                val_acc+=np.sum(wrong_pred_mask)
                num_img += np.shape(batch_labels)[0]
                cur_val_loss = val_loss / num_img
                cur_val_acc = val_acc / num_img
                val_generator_tqdm.set_description(
                    "epoch:{}/{},val_loss:{:.3f},val_acc:{:.3f}".format(epoch+1, args.epochs,cur_val_loss,cur_val_acc))
            with val_writer.as_default():
                tf.summary.scalar("train_loss-val_loss", cur_val_loss, step=epoch)
                val_writer.flush()
            with acc_writer.as_default():
                tf.summary.scalar("acc", cur_val_acc, step=epoch)
                acc_writer.flush()
            if cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                best_val_epoch = epoch+1
                best_weight_path = os.path.join(args.checkpoints, 'best_weight_{}_val_loss:{:.3f}_val_acc:{:.3f}_epoch:{}'.format(args.backbone,best_val_loss, cur_val_acc,best_val_epoch))
                model.save_weights(best_weight_path)

        cur_time = time.perf_counter()
        one_epoch_time = cur_time - epoch_start_time
        print("time elapsed: {:.3f} hour, time left: {:.3f} hour".format((cur_time-start_time)/3600,remaining_epoches*one_epoch_time/3600))

        if epoch>=1 and not open_tensorboard_url:
            open_tensorboard_url = True
            webbrowser.open(url,new=1)
    try:
        #show prediction result
        model_path = get_best_model_path(args.checkpoints)
        model.load_weights(model_path)
        wrong_pred_result = get_confusion_matrix(val_generator, model)
        print("wrong_pred_result:{}".format(wrong_pred_result))
    except:
        pass
    print("finished!")

if __name__ == '__main__':
    main(sys.argv[1:])