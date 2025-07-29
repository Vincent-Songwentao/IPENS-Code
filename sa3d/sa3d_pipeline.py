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

"""Segment Anything in 3D Pipeline"""

from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText
from nerfstudio.utils import profiler

from sa3d.sa3d_datamanager import SA3DDataManagerConfig
from sa3d.sa3d import SA3DModelConfig
from sa3d.self_prompting.sam3d import SAM3DConfig, SAM3D

"""
SWT
"""
import os
import subprocess
import cv2
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# SAM2
def gen_point_list():
    return [[[640,360],[650,380]],[[640,360],[650,380]]]
def gen_point_label_list():
    return [[1,0],[1,0]]
def gen_point_frame_list():
    return [0,1]
def gen_box_list():
    return [[[796, 119, 849, 170],[796, 119, 849, 170]],[[796, 119, 849, 170],[796, 119, 849, 170]]]
# # SAM1 ******************
# def gen_point_list():
#     return [[640,360],[650,380]],[[640,360],[650,380]]
# def gen_point_label_list():
#     return [1,0]
# def gen_point_frame_list():
#     return [0]

@dataclass
class SA3DPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""
    _target: Type = field(default_factory=lambda: SA3DPipeline)
    """target class to instantiate"""
    datamanager: SA3DDataManagerConfig = SA3DDataManagerConfig()
    """specifies the datamanager config"""
    model: SA3DModelConfig = SA3DModelConfig()
    """specifies the model config"""
    network: SAM3DConfig = SAM3DConfig()
    """specifies the segmentation model SAM3D config"""
    text_prompt: str = "the center object"
    """text prompt"""
    """
    SWT
    Add:point_prompt
    Add:BOX-PROMPT
    """
    # SAM2
    point_prompt: list[list[list[int]]] = field(default_factory=gen_point_list)
    """point prompt"""
    point_label:list[list[int]] = field(default_factory=gen_point_label_list)
    point_frame:list[int] = field(default_factory=gen_point_frame_list)
    box_prompt:list[list[list[int]]] = field(default_factory=gen_box_list)
    use_yolo: bool = field(default_factory=lambda: False)


    # SAM1
    # point_prompt: list[list[int]] = field(default_factory=gen_point_list)
    # """point prompt"""
    # point_label:list[int] = field(default_factory=gen_point_label_list)
    # point_frame:list[int] = field(default_factory=gen_point_frame_list)

class SA3DPipeline(VanillaPipeline):
    """SA3D pipeline"""

    config: SA3DPipelineConfig

    def __init__(
        self,
        config: SA3DPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        self.sam: SAM3D = config.network.setup(device=device)
        # viewer elements
        self.text_prompt_box = ViewerText(name="Text Prompt", default_value=self.config.text_prompt, cb_hook=self.text_prompt_callback)

    def text_prompt_callback(self, handle: ViewerText) -> None:
        """Callback for text prompt box, change prompt in config"""
        self.config.text_prompt = handle.value

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs mask inverse.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        """
        SWT
            next_train return 下一张图片的光线，下一个batch就是一张图片
            改变：由于dataloder打乱的dataset中图片索引序列，所以可以修改next—train函数，顺便返回image——idx。
        """

        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)

        """
        SWT
        """
        image_curr_pos,image_curr_name = self.datamanager.next_train_image_pos(step)
        demo_name = self.datamanager.train_dataset.image_filenames[image_curr_pos].parent.parent.name
        """
        SWT
            改变：在config添加点和标签
        """
        init_prompt = None if step != 0 else self.config.point_prompt
        init_label = None if step != 0 else self.config.point_label

        # SWT
        init_frame = None if step != 0 else self.config.point_frame
        init_box_prompt = None if step != 0 else self.config.box_prompt
        use_yolo = False if step != 0 else self.config.use_yolo
        #######################################################################################################
        """
        SWT
        """
        sa3d_project_dir = "/home/z790/SynologyDrive/444-Nerf/NeRF/SegmentAnythingin3D-nerfstudio-version"
        demo_dir = self.config.datamanager.data
        sam2_input_data_dir = os.path.join(sa3d_project_dir, demo_dir, "images")
        if step == 0:
            import json
            # 执行sam2,输入训练数据，输出到sa3d_sam2_output
            cmd = "python sa3d_sam2_point.py"
            subprocess_input = {
                "sam2_input_data_dir": sam2_input_data_dir ,
                "init_prompt":init_prompt,
                "init_label":init_label,
                "init_frame":init_frame,
                "init_box_prompt": None
            }
            # box_prompt
            if len(init_box_prompt) != 0:

                subprocess_input = {
                    "sam2_input_data_dir": sam2_input_data_dir,
                    "init_prompt": None,
                    "init_label": None,
                    "init_frame": init_frame,
                    "init_box_prompt":init_box_prompt
                }
            # use_yolo
            if use_yolo:
                print("use_yolo")
                subprocess_input = {
                    "sam2_input_data_dir": sam2_input_data_dir,
                    "init_prompt": None,
                    "init_label": None,
                    "init_frame": init_frame,
                    "init_box_prompt": None,
                }
            subprocess_input_json = json.dumps(subprocess_input)
            new_env = os.environ.copy()
            new_env['PATH'] = '/home/z790/anaconda3/envs/sam2/bin'  # 修改PATH环境变量
            new_env['MY_VAR'] = 'sam2'  # 添加一个新的环境变量
            devNull = open(os.devnull, 'w')
            execution_path = "/home/z790/SynologyDrive/444-Nerf/NeRF/sam2nerf/sam2/sam2"
            p = subprocess.Popen(cmd, env=new_env, cwd=execution_path, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            stdout, stderr = p.communicate(input = subprocess_input_json + '\n')

        # #######################################################################################################
        """
        SWT
            sam2读取更新
        """
        if not self.config.use_yolo:
            sam_outputs, loss_dict, metrics_dict = self.sam.get_outputs_sam2(model_outputs,demo_name,image_curr_name,init_prompt=init_prompt,init_label=init_label) # dict_keys(['sam_mask', 'prompt_points', 'sam_mask_show'])
        else:
            sam_outputs, loss_dict, metrics_dict = self.sam.get_outputs_sam2_yolo(model_outputs,demo_name,image_curr_name,init_prompt=init_prompt) # dict_keys(['sam_mask', 'prompt_points', 'sam_mask_show'])

        model_outputs.update(sam_outputs)
        return model_outputs, loss_dict, metrics_dict
    
    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state, strict=False)

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
