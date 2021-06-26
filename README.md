
# FastClassification

FastClassification is a tensorflow toolbox for class classification.  It provides a training module with various backbones and training tricks towards state-of-the-art class classification.

|  features |     |  
| ---   | --- |
|     backbone    | ResNet50,ResNet101,ResNet152, EfficientNetB0,EfficientNetB1,..., EfficientNetB7|
|     augment  | <ul><li>- [x] mixup</li><li>- [x] cutmix</li><li>- [x] rand_augment</li><li>- [x] auto_augment</li></ul>| 
|     tricks  |<ul><li>- [x] Sharpness-Aware Minimization(SAM)</li><li>- [x] progressive-resizing</li><li>- [x] lr_finder</li><li>- [x] mixed-precision</li><li>- [x] warmup</li><li>- [x] concat-max-and-average-pool</li><li>- [x] label-smoothing</li><li>- [x] dataset-sample-ratio</li></ul>           | 
|     other  | <ul><li>- [x] confusion_matrix</li><li>- [x] data sanity check</li><li>- [x] class imbalance check</li><li>- [x] Precision</li><li>- [x] recall</li><li>- [x] show wrong prediction images</li><li>- [x] tensorboard</li></ul>| 

## Update Log
[2021-06-26] 
* Add classic models and training tricks.

![chess_p5_detection.png](https://github.com/wangermeng2021/FastClassification/blob/main/images/classes_hist.png)
![chess_p5_detection.png](https://github.com/wangermeng2021/FastClassification/blob/main/images/lr_finder.png)
![chess_p5_detection.png](https://github.com/wangermeng2021/FastClassification/blob/main/images/samples.png)
![chess_p5_detection.png](https://github.com/wangermeng2021/FastClassification/blob/main/images/tensorboard.png)
![chess_p5_detection.png](https://github.com/wangermeng2021/FastClassification/blob/main/images/wrong_pred.png)

## Installation
###  1. Clone project
  ``` 
  git clone https://github.com/wangermeng2021/FastClassification.git
  cd FastClassification
  ```
###   2. Install environment
* Install tesnorflow (skip this step if it's already installed,test environment:tensorflow 2.4.0)
* Install dependencies:  `pip install -r requirements.txt`

## Training:


* Set the learning rate manually:
  ```
  python train.py --init-lr 1e-3 --progressive-resizing 224 224 --dataset-dir dataset/catdog --epochs 10 --batch-size 32 --augment cutmix --weights imagenet
  ```
* For training with Sharpness-Aware Minimization optimizer(support SAM-SGD,SAM-Adam):
  ```
  python train.py --optimizer SAM-SGD --augment baseline --init-lr 1e-3 --progressive-resizing 224 224 --dataset-dir dataset/catdog --epochs 10 --batch-size 32  --weights imagenet
  ```
* For training with lr-finder:
  ```
  python train.py --init-lr 0 --progressive-resizing 224 224 --dataset-dir dataset/catdog --epochs 10 --batch-size 32 --augment cutmix --weights imagenet
  ```
* For training with progressive-resizing:[(128,128),[224,224]]
  ```
  python train.py --progressive-resizing 128 128 224 224  --dataset-dir dataset/catdog  --epochs 10 --batch-size 32 --augment cutmix --weights imagenet
  ```  
* For training with mixed-precision:
  ```
  python train.py --mixed-precision True --init-lr 1e-3 --progressive-resizing 224 224 --dataset-dir dataset/catdog --epochs 10 --batch-size 32 --augment cutmix --weights imagenet
  ```
* For training with rand_augment:
  ```
  python train.py --augment rand_augment --init-lr 1e-3 --progressive-resizing 224 224 --dataset-dir dataset/catdog --epochs 10 --batch-size 32  --weights imagenet
  ```

  
## Tensorboard visualization:
  * Navigate to [http://0.0.0.0:6006](http://0.0.0.0:6006): you need to manually enable: "Setting"-->"Reload data" on tensorboard home page to automatically update data

## Evaluation results on toy catdog:
### (1000 pictures,train:valid=0.7:0.3,EfficientB0,epochs=10,batchsize=32) 
| model                                               | val_acc |  
|-----------------------------------------------------|---------|
| progressive-resizing(224x224)+SGD+baseline                                          |  0.987   |
| progressive-resizing(224x224)+SAM-SGD+baseline_augment                              |  0.990   |
| progressive-resizing(224x224)+SAM-SGD+mixup                                         |  0.947   |
| progressive-resizing(224x224)+SAM-SGD+cutmix                                        |  0.927   |
| progressive-resizing(224x224)+SAM-SGD+auto_augment                                  |  0.980   |
| progressive-resizing(224x224)+SAM-SGD+rand_augment                                  |  0.960   |
| progressive-resizing(224x224)+SAM-SGD+baseline+mixed-precision                      |  0.980   |
| progressive-resizing(224x224)+SAM-SGD+baseline+lr-finder                            |  0.907   |
| progressive-resizing(128x128,224x224)+SAM-SGD+baseline                              |  0.963   |
| progressive-resizing(128x128,224x224)+SAM-SGD+mixup                                 |  0.936   |


## References
* [https://github.com/wcipriano/pretty-print-confusion-matrix](https://github.com/wcipriano/pretty-print-confusion-matrix)
* [https://github.com/Jannoshh/simple-sam](https://github.com/Jannoshh/simple-sam)



