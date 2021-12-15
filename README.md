# [HiT-GAN](https://arxiv.org/pdf/2106.07631.pdf) Official TensorFlow Implementation

HiT-GAN presents a Transformer-based generator that is trained based on Generative Adversarial Networks (GANs). It achieves state-of-the-art performance for high-resolution image synthesis. Please check our [NeurIPS 2021 paper](https://arxiv.org/pdf/2106.07631.pdf) for more details.

This implementation is based on TensorFlow 2.x. We use `tf.keras` layers for building the model and use `tf.data` for our input pipeline. The model is trained using a custom training loop with `tf.distribute` on multiple TPUs/GPUs.

## Environment setup

It is recommended to run distributed training to train our model with TPUs and evaluate it with GPUs. The code is compatible with TensorFlow 2.x. See requirements.txt for all prerequisites, and you can also install them using the following command.

```
pip install -r requirements.txt
```

## ImageNet

At the first time, download ImageNet following `tensorflow_datasets` instruction from the [official guide](https://www.tensorflow.org/datasets/catalog/imagenet2012).

### Train on ImageNet

To pretrain the model on ImageNet with Cloud TPUs, first check out the [Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist) for basic information on how to use Google Cloud TPUs.

Once you have created virtual machine with Cloud TPUs, and pre-downloaded the ImageNet data for [tensorflow_datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012), please set the following enviroment variables:

```
TPU_NAME=<tpu-name>
STORAGE_BUCKET=gs://<storage-bucket>
DATA_DIR=$STORAGE_BUCKET/<path-to-tensorflow-dataset>
MODEL_DIR=$STORAGE_BUCKET/<path-to-store-checkpoints>
```

The following command can be used to train a model on ImageNet (which reflects the default hyperparameters in our paper) on TPUv2 4x4:

```
python run.py --mode=train --dataset=imagenet2012 \
  --train_batch_size=256 --train_steps=1000000 \
  --image_crop_size=128 --image_crop_proportion=0.875 \
  --save_every_n_steps=2000 \
  --latent_dim=256 --generator_lr=0.0001 \
  --discriminator_lr=0.0001 --channel_multiplier=1 \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --use_tpu=True --master=$TPU_NAME
```

To train the model on ImageNet with multiple GPUs, try the following command:

```
python run.py --mode=train --dataset=imagenet2012 \
  --train_batch_size=256 --train_steps=1000000 \
  --image_crop_size=128 --image_crop_proportion=0.875 \
  --save_every_n_steps=2000 \
  --latent_dim=256 --generator_lr=0.0001 \
  --discriminator_lr=0.0001 --channel_multiplier=1 \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --use_tpu=False --use_ema_model=False
```

Note that storing Exponential Moving Average (EMA) models is not supported with GPUs currently (`--use_ema_model=False`), which will lead to slight performance drop. Please set `train_batch_size` according to the number of GPUs for training.

### Evaluate on ImageNet

Run the following command to evaluate the model on GPUs:

```
python run.py --mode=eval --dataset=imagenet2012 \
  --eval_batch_size=128 --train_steps=1000000 \
  --image_crop_size=128 --image_crop_proportion=0.875 \
  --latent_dim=256 --channel_multiplier=1 \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --use_tpu=False --use_ema_model=True
```

This command runs models with 8 P100 GPUs. Please set `eval_batch_size` according to the number of GPUs for evaluation. Please also note that `train_steps` and `use_ema_model` should be set according to the values used for training.

## Cite

```
@inproceedings{zhao2021improved,
  title = {Improved Transformer for High-Resolution {GANs}},
  author = {Long Zhao and Zizhao Zhang and Ting Chen and Dimitris Metaxas abd Han Zhang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}
```

## Disclaimer

This is not an officially supported Google product.