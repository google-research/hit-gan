# coding=utf-8
# Copyright 2021 The HiT-GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific HiT-GAN governing permissions and
# limitations under the License.
# ==============================================================================
"""The main training and evaluation pipeline."""

from absl import app
from absl import flags
from absl import logging
from trainers import gan_trainer

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None, 'Model directory for training.')
flags.mark_flag_as_required('model_dir')

flags.DEFINE_enum('mode', None, ['train', 'eval'],
                  'Whether to perform training or evaluation.')
flags.mark_flag_as_required('mode')

flags.DEFINE_string('dataset', None, 'Dataset name.')
flags.mark_flag_as_required('dataset')

flags.DEFINE_integer('train_batch_size', 64, 'Batch size for training.')

flags.DEFINE_integer('eval_batch_size', 64, 'Batch size for evaluation.')

flags.DEFINE_integer('train_steps', 100000, 'Number of training steps.')

flags.DEFINE_string('data_dir', None, 'Data directory for the dataset.')

flags.DEFINE_integer('image_crop_size', 128, 'Size of cropped images.')

flags.DEFINE_float('image_aspect_ratio', 1.0, 'Aspect ratio of images.')

flags.DEFINE_float('image_crop_proportion', 1.0, 'Crop proportion of images.')

flags.DEFINE_bool('random_flip', True, 'Whether to use random horizontal flip.')

flags.DEFINE_integer('record_every_n_steps', 200, 'Number of steps to record.')

flags.DEFINE_integer('save_every_n_steps', 1000,
                     'Number of steps to save models.')

flags.DEFINE_integer('keep_checkpoint_max', 10,
                     'maximum number of checkpoints to keep.')

flags.DEFINE_integer('latent_dim', 256, 'Dimension of the input latents.')

flags.DEFINE_float('generator_lr', 0.0001, 'Learning rate of the generator.')

flags.DEFINE_float('discriminator_lr', 0.0001,
                   'Learning rate of the discriminator.')

flags.DEFINE_float('beta1', 0.0, 'Beta1 value of the Adam optimizer.')

flags.DEFINE_float('beta2', 0.99, 'Beta2 value of the Adam optimizer.')

flags.DEFINE_bool('use_ema_model', True,
                  'Whether to EMA weights for the generator.')

flags.DEFINE_float('ema_decay', 0.999, 'Decay value of EMA.')

flags.DEFINE_integer('ema_inital_step', 10, 'Initial step of EMA.')

flags.DEFINE_bool('use_consistency_regularization', False,
                  'Whether to use bCR for the discriminator.')

flags.DEFINE_float('consistency_regularization_cost', 10.0,
                   'Weight value of bCR.')

flags.DEFINE_string('augment_policy', 'color,translation,cutout',
                    'Policy of data augmentation.')

flags.DEFINE_enum('gan_loss_type', 'non-saturating',
                  ['non-saturating', 'hinge'],
                  'GAN loss type (non-saturating or hinge).')

flags.DEFINE_enum('grad_penalty_type', 'r1', ['r1', 'wgan'],
                  'gradieht penalty type (r1 or wgan).')

flags.DEFINE_float('grad_penalty_cost', 10.0, 'Weight of the gradieht penalty.')

flags.DEFINE_integer('channel_multiplier', 1,
                     'Factor of channel dimensions for the discriminator.')

flags.DEFINE_bool('blur_resample', True,
                  'Whether to use blur downsample for the discriminator.')

flags.DEFINE_string(
    'master', None,
    'Address/name of the TensorFlow master to use. By default, use an '
    'in-process master.')

flags.DEFINE_bool('use_tpu', True, 'Whether to run on TPU.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  strategy = None
  if FLAGS.use_tpu:
    cluster = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.master)
    tf.config.experimental_connect_to_cluster(cluster)
    topology = tf.tpu.experimental.initialize_tpu_system(cluster)
    logging.info('Topology:')
    logging.info('num_tasks: %d', topology.num_tasks)
    logging.info('num_tpus_per_task: %d', topology.num_tpus_per_task)
    strategy = tf.distribute.experimental.TPUStrategy(cluster)
  else:
    # For (multiple) GPUs.
    strategy = tf.distribute.MirroredStrategy()
    logging.info('Running using MirroredStrategy on %d replicas',
                 strategy.num_replicas_in_sync)

  generator_args = {
      'output_size': FLAGS.image_crop_size,
  }

  discriminator_args = {
      'input_size': FLAGS.image_crop_size,
      'channel_multiplier': FLAGS.channel_multiplier,
      'blur_resample': FLAGS.blur_resample
  }

  base_trainer_args = {
      'strategy': strategy,
      'model_dir': FLAGS.model_dir,
      'train_batch_size': FLAGS.train_batch_size,
      'eval_batch_size': FLAGS.eval_batch_size,
      'dataset': FLAGS.dataset,
      'train_steps': FLAGS.train_steps,
      'data_dir': FLAGS.data_dir,
      'image_crop_size': FLAGS.image_crop_size,
      'image_aspect_ratio': FLAGS.image_aspect_ratio,
      'image_crop_proportion': FLAGS.image_crop_proportion,
      'random_flip': FLAGS.random_flip,
      'record_every_n_steps': FLAGS.record_every_n_steps,
      'save_every_n_steps': FLAGS.save_every_n_steps,
      'keep_checkpoint_max': FLAGS.keep_checkpoint_max
  }

  gan_trainer_args = {
      'latent_dim': FLAGS.latent_dim,
      'generator_lr': FLAGS.generator_lr,
      'discriminator_lr': FLAGS.discriminator_lr,
      'beta1': FLAGS.beta1,
      'beta2': FLAGS.beta2,
      'use_ema_model': FLAGS.use_ema_model,
      'ema_decay': FLAGS.ema_decay,
      'ema_inital_step': FLAGS.ema_inital_step,
      'use_consistency_regularization': FLAGS.use_consistency_regularization,
      'consistency_regularization_cost': FLAGS.consistency_regularization_cost,
      'augment_policy': FLAGS.augment_policy,
      'gan_loss_type': FLAGS.gan_loss_type,
      'grad_penalty_type': FLAGS.grad_penalty_type,
      'grad_penalty_cost': FLAGS.grad_penalty_cost
  }

  trainer = gan_trainer.GANTrainer(generator_args, discriminator_args,
                                   **gan_trainer_args, **base_trainer_args)
  trainer.build()

  if FLAGS.mode == 'train':
    trainer.train()
  elif FLAGS.mode == 'eval':
    trainer.evaluate()
  else:
    raise ValueError('Trainer mode {} not supported'.format(FLAGS.mode))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  # For outside compilation of summaries on TPU.
  tf.config.set_soft_device_placement(True)
  app.run(main)
