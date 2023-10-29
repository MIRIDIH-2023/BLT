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

"""Retrieves information about the available datasets."""

import enum

from . import coco_info
from . import magazine_info
from . import publaynet_info
from . import rico_info
from . import miri_info
from . import categorized_info


@enum.unique
class DatasetName(enum.Enum):
  COCO = "COCO"
  RICO = "RICO"
  PUBLAYNET = "PubLayNet"
  MAGAZINE = "MAGAZINE"
  MIRI = "MIRI"
  CATEGORIZED = "CATEGORIZED"


def get_number_classes(dataset_name):
  """Retrieves the number of labels for a given dataset."""
  if dataset_name == DatasetName.RICO:
    return rico_info.NUMBER_LABELS
  elif dataset_name == DatasetName.PUBLAYNET:
    return publaynet_info.NUMBER_LABELS
  elif dataset_name == DatasetName.MAGAZINE:
    return magazine_info.NUMBER_LABELS
  elif dataset_name == DatasetName.COCO:
    return coco_info.NUMBER_LABELS
  elif dataset_name == DatasetName.MIRI:
    return miri_info.NUMBER_LABELS
  elif dataset_name == DatasetName.CATEGORIZED:
    return categorized_info.NUMBER_LABELS 
  else:
    raise ValueError(f"Unknown dataset '{dataset_name}'")


def get_id_to_label_map(dataset_name):
  """Retrieves the id to label map for a given dataset."""
  if dataset_name == DatasetName.RICO:
    return rico_info.ID_TO_LABEL
  elif dataset_name == DatasetName.PUBLAYNET:
    return publaynet_info.ID_TO_LABEL
  elif dataset_name == DatasetName.MAGAZINE:
    return magazine_info.ID_TO_LABEL
  elif dataset_name == DatasetName.COCO:
    return coco_info.ID_TO_LABEL
  elif dataset_name == DatasetName.MIRI:
    return miri_info.ID_TO_LABEL
  elif dataset_name == DatasetName.CATEGORIZED:
    return categorized_info.ID_TO_LABEL
  else:
    raise ValueError(f"Unknown dataset '{dataset_name}'")

def get_label_to_id_map(dataset_name):
  """Retrieves the label to id map for a given dataset."""
  if dataset_name == DatasetName.RICO:
    return rico_info.LABEL_TO_ID_
  elif dataset_name == DatasetName.PUBLAYNET:
    return publaynet_info.LABEL_TO_ID_
  elif dataset_name == DatasetName.MAGAZINE:
    return magazine_info.LABEL_TO_ID_
  elif dataset_name == DatasetName.COCO:
    return coco_info.LABEL_TO_ID_
  elif dataset_name == DatasetName.MIRI:
    return miri_info.LABEL_TO_ID_
  elif dataset_name == DatasetName.CATEGORIZED:
    return categorized_info.LABEL_TO_ID_
  else:
    raise ValueError(f"Unknown dataset '{dataset_name}'")

def get_label_name(dataset_name):
  """Retrieves the label to id map for a given dataset."""
  if dataset_name == DatasetName.RICO:
    return rico_info.LABEL_NAMES
  elif dataset_name == DatasetName.PUBLAYNET:
    return publaynet_info.LABEL_NAMES
  elif dataset_name == DatasetName.MAGAZINE:
    return magazine_info.LABEL_NAMES
  elif dataset_name == DatasetName.COCO:
    return coco_info.LABEL_NAMES
  elif dataset_name == DatasetName.MIRI:
    return miri_info.LABEL_NAMES
  elif dataset_name == DatasetName.CATEGORIZED:
    return categorized_info.TAG_NAMES
  else:
    raise ValueError(f"Unknown dataset '{dataset_name}'")