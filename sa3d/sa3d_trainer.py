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
Code to train model, in order to skip when loss is none.
"""
# SWT
# 并行化写入图片
from concurrent.futures import ThreadPoolExecutor
import cv2
from pathlib import PosixPath
#########################################
from dataclasses import dataclass, field
import functools
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch
import imageio
from numpy.ma.core import absolute
# from lxml.parser import filename
from rich.console import Console
from nerfstudio.utils import profiler
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.viewer.server.viewer_elements import ViewerButton
from nerfstudio.utils import profiler, writer
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter

CONSOLE = Console(width=120)

TRAIN_INTERATION_OUTPUT = Tuple[  # pylint: disable=invalid-name
    torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
]

@dataclass
class SA3DTrainerConfig(TrainerConfig):
    """Configuration for the SA3DTrainer."""
    # steps_per_save: int = 10
    steps_per_save: int = 50
    """Number of steps between saves."""
    steps_per_eval_batch: int = 50000
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 50000
    """Number of steps between single eval images."""
    _target: Type = field(default_factory=lambda: SA3DTrainer)

"""

trainer setup 
包括：
pipeline
optimizers
_load_checkpoint
writers/profilers
"""

class SA3DTrainer(Trainer):
    """Trainer for Segment Anything in 3D"""

    def __init__(self, config: SA3DTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:

        super().__init__(config, local_rank, world_size)

        # reset button
        self.reset_button = ViewerButton(name="Reset Button", cb_hook=self.reset_callback)
        # Train visualization
        self.vis_variables = ['sam_mask_show']
        self.train_vis = {}
        for k in self.vis_variables:
            self.train_vis.update({k: []})

    def reset_callback(self, handle: ViewerButton) -> None:
        """Reset the model to the original checkpoint"""
        
        # load checkpoint
        self._load_checkpoint()

    def save_visualzation(self):
        for k in self.vis_variables:
            self.train_vis[k] = np.stack(self.train_vis[k])
            save_dir = Path(self.base_dir / f"train_vis")
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(save_dir / f"{k}.mp4", self.train_vis[k], fps=6, quality=8)

    def update_visualzation(self, outputs):
        for k in self.vis_variables:
            if isinstance(outputs[k], np.ndarray):
                output = outputs[k]
            elif isinstance(outputs[k], torch.tensor):
                output = outputs[k].detach().cpu().numpy()
            else:
                raise ValueError('Unknown value type for {}'.format(outputs[k]))
            self.train_vis[k].append(output)


    """
    SWT
    加了个 pipeline_state_dict = {k: v for k, v in self.pipeline.state_dict().items() if "sam." not in k}
    """
    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers
        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        pipeline_state_dict = {k: v for k, v in self.pipeline.state_dict().items() if "sam." not in k}
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else pipeline_state_dict,
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
    """
    SWT
        加入了self.mask_view_counts = torch.zeros_like(self.pipeline.model.mask_fields.mask_grids.params.data)
    """
    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        camera_optimizer_config = self.config.pipeline.datamanager.camera_optimizer
        if camera_optimizer_config is not None and camera_optimizer_config.mode != "off":
            assert camera_optimizer_config.param_group not in optimizer_config
            optimizer_config[camera_optimizer_config.param_group] = {
                "optimizer": camera_optimizer_config.optimizer,
                "scheduler": camera_optimizer_config.scheduler,
            }

        self.mask_view_counts = torch.zeros_like(self.pipeline.model.mask_fields.mask_grids.params.data)

        return Optimizers(optimizer_config, param_groups)

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        # TODO: adjust loss according to view counts
        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            pipe_outputs, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            self.update_visualzation(pipe_outputs)
            if loss_dict['mask'] is None:
                return 0., loss_dict, metrics_dict
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        # leverage view count weights
        with torch.no_grad():
            self.pipeline.model.mask_fields.mask_grids.params.data *= self.mask_view_counts
            prev_mask_grids = self.pipeline.model.mask_fields.mask_grids.params.data.detach().clone()
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)
        with torch.no_grad():
            self.mask_view_counts += (self.pipeline.model.mask_fields.mask_grids.params.data != prev_mask_grids)
            self.pipeline.model.mask_fields.mask_grids.params.data /= (self.mask_view_counts + 1e-8)
        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad
                    total_grad += grad

            metrics_dict["Gradients/Total"] = total_grad

        self.grad_scaler.update()
        self.optimizers.scheduler_step_all(step)
        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
            self.base_dir / "dataparser_transforms.json"
        )

        """
        SWT
        """
        filenames = self.pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames
        def save_train_dataset(filenames,idx):
            # 读取图像
            file = filenames[idx]
            image_path = str(file) #sa3d_data/demo1/images/0114.jpg
            image_name = file.name # xxxx.jpg type=str
            images_path = str(file.parent) # sa3d_data/demo1/images
            demo_path = str(file.parent.parent) # sa3d_data/demo1
            sa3d_project_dir = "/home/z790/SynologyDrive/444-Nerf/NeRF/SegmentAnythingin3D-nerfstudio-version"
            absolute_image_path = os.path.join(sa3d_project_dir, image_path)
            img = cv2.imread(absolute_image_path)
            # 保存图像
            train_dir = os.path.join(sa3d_project_dir, demo_path, "train")
            if not os.path.exists(train_dir):
                try:
                    os.makedirs(train_dir, exist_ok=True)
                except OSError as error:
                    print(f"Failed to create folder '{train_dir}'. {error}")
            train_image_path = os.path.join(train_dir,image_name) # /home/z790/...nerfstudio-version/sa3d_data/demo1/train
            cv2.imwrite(train_image_path, img)

        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = [executor.submit(save_train_dataset, filenames,idx) for idx in
                       range(0, self.pipeline.datamanager.train_dataset.__len__())]
        for future in futures:
            future.result()
        self._init_viewer_state()
        """
        SWT 
            计算时间 
        """
        start_time = time.perf_counter()

        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            # num_iterations = self.config.max_num_iterations
            num_iterations = self.pipeline.datamanager.len_image_batch # 97
            self._start_step = step = 0
            for step in range(self._start_step, self._start_step + num_iterations):
                while not self.is_training:
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()
                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.pipeline.datamanager.get_train_rays_per_batch() / train_t.duration,
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        end_time = time.perf_counter()
        # 计算并打印执行时间
        execution_time = end_time - start_time

        # save checkpoint at the end of training
        self.save_checkpoint(step)
        self.save_visualzation()

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        CONSOLE.rule()
        CONSOLE.print("[bold green]:tada: :tada: :tada: Training Finished :tada: :tada: :tada:", justify="center")
        if not self.config.viewer.quit_on_train_completion:
            CONSOLE.print("Use ctrl+c to quit    "+str(execution_time), justify="center")
            while True:
                time.sleep(0.01)
    """
    SWT
    注释了两行self
    """
    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir: Path = self.config.load_dir
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            # self.optimizers.load_optimizers(loaded_state["optimizers"])
            # self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"done loading checkpoint from {load_path}")
        else:
            CONSOLE.print("No checkpoints to load, training from scratch")