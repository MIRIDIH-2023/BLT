# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main file for running the example.

This file is intentionally kept short. The majority for logic is in libraries
than can be easily tested and imported in Colab.
"""

from absl import app
from absl import flags
from absl import logging

# Required import to setup work units when running through XManager.
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf
from trainers import bert_layout_trainer
from trainers import transformer_trainer
# import cv2
import plot_layout
import numpy as onp
import json
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.mark_flags_as_required(["config", "workdir"])
flags.DEFINE_string("mode", "train", "job status")

# Flags --jax_backend_target and --jax_xla_backend are available through JAX.


def get_trainer_cls(config, workdir):
  """Get model."""
  if config.model_class == "transformer":
    return transformer_trainer.TransformerTrainer(config, workdir)
  elif config.model_class == "bert_layout":
    return bert_layout_trainer.BERTLayoutTrainer(config, workdir)
  else:
    raise NotImplementedError(f"{config.model_class} is not Implemented")
  
def create_file_name(conditional, exp):
    exp = exp.split('/')[1]
    if conditional == "a":
        file_name = "report_{0}_{1}.json".format(exp, conditional)
    elif conditional == "a+s":
        file_name = "report_{0}_{1}.json".format(exp, conditional.replace('+', '_'))
    else :
        file_name = "report_{0}_{1}.json".format(exp, conditional)

    return file_name


def main(argv):
  del argv

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  if FLAGS.jax_backend_target:
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
    jax_xla_backend = ("None" if FLAGS.jax_xla_backend is None else
                       FLAGS.jax_xla_backend)
    logging.info("Using JAX XLA backend %s", jax_xla_backend)

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())

  platform.work_unit().set_task_status(f"process_index: {jax.process_index()}, "
                                       f"process_count: {jax.process_count()}")
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, "workdir")
  trainer = get_trainer_cls(FLAGS.config, FLAGS.workdir)
  if FLAGS.mode == "train":
    trainer.train()
  elif FLAGS.mode == "test":
    while(1) :
      idx = None
      idx = int(input("enter the index number (or -1 to quit): "))
      if idx == -1 : break

      iteration = int(input("enter iter: "))
      condition = input("enter decode condition (a or a+s): ")

      generated_samples, real_samples, image_link = trainer.test_with_backgroundImage(conditional=condition, iterative_nums=[iteration, iteration, iteration], idx=idx)
      plot_layout.plot_sample_with_PIL(
        data=onp.array(generated_samples[0][-1]),
        workdir=FLAGS.workdir,
        dataset_type="CATEGORIZED",
        border_size= 1,
        thickness= 6,
        im_type=f"{idx}_iter{iteration}_infer",
        idx=idx, 
        image_link=image_link,
        conditional=condition,
        composition=FLAGS.config.composition)
      print()
      plot_layout.plot_sample_with_PIL(
        data=onp.array(real_samples[0]),
        workdir=FLAGS.workdir,
        dataset_type="CATEGORIZED",
        border_size= 1,
        thickness= 6,
        im_type=f"{idx}_iter{iteration}_real",
        idx=idx,
        image_link=image_link,
        conditional=condition,
        composition=FLAGS.config.composition)
  elif FLAGS.mode == "eval":
    total_iteration = int(input("enter total iteration: "))
    condition = input("enter decode condition (a or a+s): ")
    report = []
    for iteration in range(1, total_iteration+1) :
      generated_samples, real_samples = trainer.test(conditional=condition, iterative_nums=[iteration, iteration, iteration])
      result = trainer.evaluate_metrics(generated_samples=generated_samples,
                                        real_samples=real_samples,
                                        conditional=condition,
                                        composition=FLAGS.config.composition)

      report.append(result)

    report_path = "/home/work/increased_en_data/BLT/report"
    file_name = create_file_name(condition, FLAGS.workdir)

    # assert file_name != None
    data = { "report" : report }

    file_path = os.path.join(report_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
      json.dump(data, file, indent="\t")
  else:
    raise NotImplementedError


if __name__ == "__main__":
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)
