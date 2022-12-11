# Improving SogCLR with Intermediate-layer Loss and New Data Augmentation

This repo stores the codes for TAMU CSCE 689 optimization for machine learning course project. In this project, we propose two strategies to improve a self-supervised learning framework [SogCLR](https://arxiv.org/abs/2202.12387), including intermediate loss and new data augmentations. In this repo, you can run the improved SogCLR on CIFAR-10 and CIFAR-100 datasets. We recommend to run codes on a GPU-enabled environment, e.g., [Google Colab](https://colab.research.google.com/). You can run our codes in this [Google Colab Notebook](https://colab.research.google.com/drive/1WtaHmpwsMgLMnzcLeXx_YliOoXEb70RT?usp=sharing)


## Installation

### Required packages

- PyTorch
- Torchvision

### git clone
```bash
git clone https://github.com/lyzustc/CSCE689_Opt4ML_course_project
```

## Training  
Below are examples for running SogCLR pre-training with our proposed strategies of a ResNet-50 model on CIFAR-10 on a single GPU. The first time you run the scripts, datasets will be automatically downloaded to `/data/`. 
- By default, we use linear learning rate scaling, e.g., $\text{LearningRate}=1.0\times\text{BatchSize}/256$, [LARS](https://arxiv.org/abs/1708.03888) optimizer and a weight decay of 1e-4. For temperature parameter $\tau$, we use a fixed value of 0.3. For SogCLR, gamma (γ in the paper) is an additional parameter for maintaining moving average estimator, the default value is 0.9. 
- By default, `CIFAR-10` is used for experiments. To pretrain on CIFAR-100, you can set `--data_name cifar100`. In this repo, only `CIFAR-10/CIFAR-100` is supported, however, you can modify the dataloader to support other datasets.


**CIFAR**

We use batch size of 64 and train 400 epochs for pretraining. You can also increase the number of workers to accelerate the training speed.

Execute the following command to run SogCLR pre-training with intermediate-layer loss.
```bash
python train_ill.py \
  --lr=1.0 --learning-rate-scaling=sqrt \
  --epochs=400 --batch-size=64 \
  --loss_type dcl \
  --use_new_aug False \
  --gamma 0.9 \
  --workers 32 \
  --wd=1e-4 \
  --data_name cifar10 \
  --save_dir ./cifar10_ill/ \
  --print-freq 1000
```

Execute the following command to run SogCLR pre-training with new data augmentation.
```bash
python train.py \
  --lr=1.0 --learning-rate-scaling=sqrt \
  --epochs=400 --batch-size=64 \
  --loss_type dcl \
  --use_new_aug True \
  --jig_prob 0.1 \
  --gauss_prob 0.01 \
  --solar_prob 0.1 \
  --jig_num 2 \
  --gamma 0.9 \
  --workers 32 \
  --wd=1e-4 \
  --data_name cifar10 \
  --save_dir ./cifar_nda/ \
  --print-freq 1000
```

Execute the following command to run SogCLR pre-training with intermediate-layer loss and new data augmentation.
```bash
python train_ill.py \
  --lr=1.0 --learning-rate-scaling=sqrt \
  --epochs=400 --batch-size=64 \
  --loss_type dcl \
  --use_new_aug True \
  --jig_prob 0.1 \
  --gauss_prob 0.01 \
  --solar_prob 0.1 \
  --jig_num 2 \
  --gamma 0.9 \
  --workers 32 \
  --wd=1e-4 \
  --data_name cifar10 \
  --save_dir ./cifar10_ill_nda/ \
  --print-freq 1000
```

## Linear evaluation
By default, we use momentum-SGD without weight decay and a batch size of 1024 for linear evaluation on on frozen features/weights. In this stage, it runs 90 epochs for training.

**CIFAR**

```bash
python lincls.py \
  --workers 32 \
  --data_name cifar10 \
  --save_dir ./cifar10_ill/
```