# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Segment Anything in 3D Datamanager.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type
import torch
from pyasn1_modules.rfc1902 import Integer
from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from tensorboard.plugins.image.summary import image
import os

CONSOLE = Console(width=120)

@dataclass
class SA3DDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the SA3DDataManager."""

    _target: Type = field(default_factory=lambda: SA3DDataManager)
    patch_size: int = 1
    """Size of patch to sample from. If >1, patch-based sampling will be used."""

class SA3DDataManager(VanillaDataManager):
    """
    Manages data operations specific to the SA3D (Segment Anything in 3D) pipeline,
    extending the functionality of the VanillaDataManager to handle training datasets
    with customized loading strategies and ray generation for 3D scene understanding tasks.

    Attributes:
        config (SA3DDataManagerConfig): Configuration object specifying parameters
            for setting up the training data, including number of images to sample,
            repetitions, and other optimizations.

    Methods:
        setup_train:
            Sets up the training dataset by initializing data loaders, pixel samplers,
            camera optimizers, and ray generators according to the provided configuration.

        next_train:
            Yields the next batch of rays and associated data for training, cycling
            through the pre-fetched image batch to maintain a consistent sampling pattern.

        next_train_image_pos:
            Returns the positional index of the current image within the training dataset's folder structure.
    """

    config: SA3DDataManagerConfig

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )
        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)
        # print("self.image_batch",self.image_batch)
        # print(self.image_batch['image_idx']) # 种子固定
        self.len_image_batch = len(self.image_batch["image"])


    """
    SWT
    改变next_train step为train_count-1来解决输出单视角问题
    """

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        # current_index = torch.tensor([step % self.len_image_batch]) # 0-96
        current_index = torch.tensor([(self.train_count-1) % self.len_image_batch]) # 0-96

        # get current camera, include camera transforms from original optimizer
        camera_transforms = self.train_camera_optimizer(current_index)
        current_camera = self.train_dataparser_outputs.cameras[current_index].to(self.device)
        current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)
        batch = {"image": self.image_batch["image"][current_index]} # [H, W, C]
        return current_ray_bundle, batch



    def next_train_image_pos(self, step: int)-> Tuple[Integer,str]:
        current_index = torch.tensor([step % self.len_image_batch])  # 0-96
        """
        SWT
        """
        image_cur = self.train_dataset.image_filenames[current_index]
        demo_name = str(image_cur.parent.parent.name)
        sa3d_data_dir = "/home/z790/SynologyDrive/444-Nerf/NeRF/SegmentAnythingin3D-nerfstudio-version/sa3d_data"
        absolute_image_path = os.path.join(sa3d_data_dir, demo_name,"train")
        # 将文件名转换为绝对路径，并检查目标文件是否存在
        full_paths = [os.path.join(absolute_image_path, file) for file in sorted(os.listdir(absolute_image_path))]
        image_cur_path = os.path.join(absolute_image_path, image_cur.name)
        if image_cur_path in full_paths:
            # 获取目标文件在列表中的索引位置，索引从0开始
            position = full_paths.index(image_cur_path)
            # print(f"The file '{image_cur.name}' is at position {position} in the folder.")
        else:
            print(f"The file '{image_cur.name}' does not exist in the folder.")

        return position,image_cur.name
