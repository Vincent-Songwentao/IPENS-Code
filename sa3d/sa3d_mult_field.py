try:
    import tinycudann as tcnn
except ImportError:
    pass
import torch
from typing import Dict, Optional, Tuple, Type
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from dataclasses import dataclass, field
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field

class MultiObjectMaskField(torch.nn.Module):
    def __init__(self, num_obj, num_levels, features_per_level, log2_hashmap_size, base_res, growth_factor):
        super().__init__()
        self.num_obj = num_obj  # 目标数量

        # 初始化多个编码器，每个目标一个
        self.mask_grids = torch.nn.ModuleList([
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": num_levels,
                    "n_features_per_level": features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_res,
                    "per_level_scale": growth_factor,
                    "interpolation": "Linear"
                }
            )
            for _ in range(num_obj)
        ])
        self.mask_grids.requires_grad = True
    def forward(self, positions_flat, obj_id):
        """
        使用指定目标 ID 的编码器对输入点进行编码。

        参数:
            positions_flat: Tensor，形状为 [num_points, 3]，输入点坐标。
            obj_id: int，目标 ID。

        返回:
            编码后的特征，形状为 [num_points, n_levels * features_per_level]。
        """
        assert 0 <= obj_id < self.num_obj, f"Invalid obj_id: {obj_id}. Must be in range [0, {self.num_obj-1}]"
        return self.mask_grids[obj_id](positions_flat)

    def get_multi_object_mask_weights(self, positions_flat, weights, num_obj):
        """
        计算多目标的 mask_weights。

        参数:
            positions_flat: [num_rays * num_samples, 3]，输入的三维坐标。
            weights: [num_rays, num_samples]，射线权重。
            num_obj: int，目标数量。

        返回:
            mask_weights: [num_rays, num_obj] 的多目标权重。
        """
        # num_rays, num_samples = weights.shape
        mask_weights_list = []

        for obj_id in range(num_obj):
            # 使用独立的编码器计算 mask_weights
            mask_weights = self.forward(positions_flat, obj_id).reshape(*weights.shape[:2], -1)
            mask_weights_list.append(mask_weights)

        # 将所有目标的 mask_weights 堆叠
        mask_weights = torch.stack(mask_weights_list, dim=-1)  # [num_rays, num_samples, num_levels * features_per_level, num_obj]
        return mask_weights

