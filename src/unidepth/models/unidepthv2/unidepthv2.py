"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import importlib
from copy import deepcopy
from math import ceil
import warnings
import torchvision.utils as vutils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from unidepth.models.unidepthv2.decoder import Decoder
from unidepth.utils.camera import BatchCamera, Camera, Pinhole
from unidepth.utils.constants import (IMAGENET_DATASET_MEAN,
                                      IMAGENET_DATASET_STD)
from unidepth.utils.distributed import is_main_process
from unidepth.utils.misc import (first_stack, get_params, last_stack, match_gt,
                                 match_intrinsics, max_stack, mean_stack,
                                 softmax_stack)
from src.misc.utils import vis_depth_map

STACKING_FNS = {
    "max": max_stack,
    "mean": mean_stack,
    "first": first_stack,
    "last": last_stack,
    "softmax": softmax_stack,
}


def get_paddings(original_shape, aspect_ratio_range):
    # Original dimensions
    H_ori, W_ori = original_shape
    orig_aspect_ratio = W_ori / H_ori

    # Determine the closest aspect ratio within the range
    min_ratio, max_ratio = aspect_ratio_range
    target_aspect_ratio = min(max_ratio, max(min_ratio, orig_aspect_ratio))

    if orig_aspect_ratio > target_aspect_ratio:  # Too wide
        W_new = W_ori
        H_new = int(W_ori / target_aspect_ratio)
        pad_top = (H_new - H_ori) // 2
        pad_bottom = H_new - H_ori - pad_top
        pad_left, pad_right = 0, 0
    else:  # Too tall
        H_new = H_ori
        W_new = int(H_ori * target_aspect_ratio)
        pad_left = (W_new - W_ori) // 2
        pad_right = W_new - W_ori - pad_left
        pad_top, pad_bottom = 0, 0

    return (pad_left, pad_right, pad_top, pad_bottom), (H_new, W_new)


def get_resize_factor(original_shape, pixels_range, shape_multiplier=14):
    # Original dimensions
    H_ori, W_ori = original_shape
    n_pixels_ori = W_ori * H_ori

    # Determine the closest number of pixels within the range
    min_pixels, max_pixels = pixels_range
    target_pixels = min(max_pixels, max(min_pixels, n_pixels_ori))

    # Calculate the resize factor
    resize_factor = (target_pixels / n_pixels_ori) ** 0.5
    new_width = int(W_ori * resize_factor)
    new_height = int(H_ori * resize_factor)
    new_height = ceil(new_height / shape_multiplier) * shape_multiplier
    new_width = ceil(new_width / shape_multiplier) * shape_multiplier

    return resize_factor, (new_height, new_width)


def _postprocess(tensor, shapes, paddings, interpolation_mode="bilinear"):
    # interpolate to original size
    tensor = F.interpolate(
        tensor, size=shapes, mode=interpolation_mode, align_corners=False
    )

    # remove paddings
    pad1_l, pad1_r, pad1_t, pad1_b = paddings
    tensor = tensor[..., pad1_t : shapes[0] - pad1_b, pad1_l : shapes[1] - pad1_r]
    return tensor


def _postprocess_intrinsics(K, resize_factors, paddings): #뎁스만 필요하면 필요없음
    batch_size = K.shape[0]
    K_new = K.clone()

    for i in range(batch_size):
        scale = resize_factors[i]
        pad_l, _, pad_t, _ = paddings[i]

        K_new[i, 0, 0] /= scale  # fx
        K_new[i, 1, 1] /= scale  # fy
        K_new[i, 0, 2] /= scale  # cx
        K_new[i, 1, 2] /= scale  # cy

        K_new[i, 0, 2] -= pad_l  # cx
        K_new[i, 1, 2] -= pad_t  # cy

    return K_new


class UniDepthV2(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="UniDepth",
    repo_url="https://github.com/lpiccinelli-eth/UniDepth",
    tags=["monocular-metric-depth-estimation"],
):
    def __init__(
        self,
        config,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.build(config)
        #self.build_losses(config)

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16)
    def infer(
        self,
        rgb: torch.Tensor,
        camera: torch.Tensor | Camera | None = None,
        normalize=True,
    ):
        ratio_bounds = self.shape_constraints["ratio_bounds"]
        pixels_bounds = [
            self.shape_constraints["pixels_min"],
            self.shape_constraints["pixels_max"],
        ]
        if hasattr(self, "resolution_level"):
            assert (
                self.resolution_level >= 0 and self.resolution_level < 10
            ), "resolution_level should be in [0, 10)"
            pixels_range = pixels_bounds[1] - pixels_bounds[0]
            interval = pixels_range / 10
            new_lowbound = self.resolution_level * interval + pixels_bounds[0]
            new_upbound = (self.resolution_level + 1) * interval + pixels_bounds[0]
            pixels_bounds = (new_lowbound, new_upbound)
        else:
            warnings.warn("!! self.resolution_level not set, using default bounds !!")

        # houskeeping on cpu/cuda and batchify
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if camera is not None:
            if isinstance(camera, torch.Tensor):
                assert (
                    camera.shape[-1] == 3 and camera.shape[-2] == 3
                ), "camera tensor should be of shape (..., 3, 3): assume pinhole"
                camera = Pinhole(K=camera)
            camera = BatchCamera.from_camera(camera)
            camera = camera.to(self.device)
        B, _, H, W = rgb.shape

        rgb = rgb.to(self.device)
        if camera is not None:
            camera = camera.to(self.device)

        # preprocess
        paddings, (padded_H, padded_W) = get_paddings((H, W), ratio_bounds)
        (pad_left, pad_right, pad_top, pad_bottom) = paddings
        resize_factor, (new_H, new_W) = get_resize_factor(
            (padded_H, padded_W), pixels_bounds
        )
        # -> rgb preprocess (input std-ized and resized)
        if normalize:
            rgb = TF.normalize(
                rgb.float() / 255.0,
                mean=IMAGENET_DATASET_MEAN,
                std=IMAGENET_DATASET_STD,
            )
        rgb = F.pad(rgb, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        rgb = F.interpolate(
            rgb, size=(new_H, new_W), mode="bilinear", align_corners=False
        )
        # -> camera preprocess
        if camera is not None:
            camera = camera.crop(
                left=-pad_left, top=-pad_top, right=-pad_right, bottom=-pad_bottom
            )
            camera = camera.resize(resize_factor)

        #try:
        #    # ImageNet 정규화(normalize)를 되돌리기 위한 mean/std 텐서
        #    mean = torch.tensor(IMAGENET_DATASET_MEAN, device=self.device).view(1, 3, 1, 1)
        #    std = torch.tensor(IMAGENET_DATASET_STD, device=self.device).view(1, 3, 1, 1)
        #    
        #    # De-normalize: (normalized * std) + mean
        #    rgb_to_save = rgb.clone() * std + mean
        #    
        #    # [0, 1] 범위로 클램핑
        #    rgb_to_save = torch.clamp(rgb_to_save, 0, 1)
#
        #    vutils.save_image(
        #        rgb_to_save, 
        #        "debug_uni_input.png", # ⬅️ 저장될 파일명
        #        nrow=B # 배치의 모든 이미지를 한 줄로
        #    )
        #except Exception as e:
        #    print(f"[Debug] Failed to save pre_encode_input image: {e}")

        # run model
        _, model_outputs = self.encode_decode(
            inputs={"image": rgb, "camera": camera}, image_metas=[]
        )

        ## ⬇️ [디버깅 코드 1] encode_decode 직후 뎁스 맵 저장 ⬇️
        #try:
        #    # model_outputs["points"]는 [B, 3, H_feat, W_feat] 형태 (xyz 맵)
        #    # 우리는 Z축이 뎁스이므로, 마지막 채널 (index -1)을 가져옵니다.
        #    # 이 뎁스 맵은 아직 _postprocess 되기 전, 모델의 특징 맵 해상도입니다.
        #    pre_postprocess_depth_raw = model_outputs["points"][:, -1:].squeeze(1) # [B, H_feat, W_feat]
#
        #    # 시각화를 위해 컬러맵으로 변환
        #    colored_depth_maps_pre = vis_depth_map(pre_postprocess_depth_raw)
#
        #    vutils.save_image(
        #        colored_depth_maps_pre, 
        #        "debug_depth_check_encode_decode.png", # ⬅️ 저장될 파일명 1
        #        nrow=model_outputs["points"].shape[0] # 배치 사이즈
        #    )
        #except Exception as e:
        #    print(f"[Debug] Failed to save depth_raw_from_encode_decode: {e}")
        ## ⬆️ [디버깅 코드 1 끝] ⬆️

        # collect outputs
        out = {}

        points = _postprocess(
            model_outputs["points"],
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode=self.interpolation_mode,
        )

        out["depth"] = points[:, -1:]
        
        ## ⬇️ [디버깅 코드 추가] 
        ## 최종 뎁스 맵 'out["depth"]'를 시각화하여 저장합니다.
        #try:
        #    # vis_depth_map은 [V, H, W] 또는 [H, W]를 받으므로 squeeze
        #    # out["depth"]는 [B*V, 1, H_out, W_out] 형태
        #    depth_to_visualize = out["depth"].squeeze(1) # [B*V, H, W]
        #    
        #    # 컬러맵 이미지로 변환 [B*V, 3, H, W]
        #    colored_depth_maps = vis_depth_map(depth_to_visualize)
#
        #    vutils.save_image(
        #        colored_depth_maps, 
        #        "debug_uni_output_depth.png", # ⬅️ 저장될 파일명
        #        nrow=B # 배치의 모든 이미지를 한 줄로
        #    )
        #except Exception as e:
        #    print(f"[Debug] Failed to save output depth image: {e}")
        ## --- [디버깅 코드 끝] ---

        #out["points"] = points
        #out["rays"] = rays / torch.norm(rays, dim=1, keepdim=True).clip(min=1e-5)
        out["depth_features"] = model_outputs["depth_features"]
        return out

    def encode_decode(self, inputs, image_metas=[]):
        B, _, H, W = inputs["image"].shape

        if inputs.get("camera", None) is not None:
            inputs["rays"] = inputs["camera"].get_rays(shapes=(B, H, W))

        features, tokens = self.pixel_encoder(inputs["image"])
        inputs["features"] = [
            self.stacking_fn(features[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        inputs["tokens"] = [
            self.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]

        outputs = self.pixel_decoder(inputs, image_metas)
        outputs["rays"] = rearrange(outputs["rays"], "b (h w) c -> b c h w", h=H, w=W)
        pts_3d = outputs["rays"] * outputs["radius"]
        outputs.update({"points": pts_3d, "depth": pts_3d[:, -1:]})

        return inputs, outputs

    def load_pretrained(self, model_file):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dict_model = torch.load(model_file, map_location=device, weights_only=False)
        if "model" in dict_model:
            dict_model = dict_model["model"]
        dict_model = {k.replace("module.", ""): v for k, v in dict_model.items()}
        info = self.load_state_dict(dict_model, strict=False)
        if is_main_process():
            print(
                f"Loaded from {model_file} for {self.__class__.__name__} results in:",
                info,
            )


    @property
    def device(self):
        return next(self.parameters()).device

    def build(self, config):
        mod = importlib.import_module("unidepth.models.encoder")
        pixel_encoder_factory = getattr(mod, config["model"]["pixel_encoder"]["name"])
        pixel_encoder_config = {
            **config["training"],
            **config["model"]["pixel_encoder"],
            **config["data"],
        }
        pixel_encoder = pixel_encoder_factory(pixel_encoder_config)

        config["model"]["pixel_encoder"]["patch_size"] = (
            14 if "dino" in config["model"]["pixel_encoder"]["name"] else 16
        )
        pixel_encoder_embed_dims = (
            pixel_encoder.embed_dims
            if hasattr(pixel_encoder, "embed_dims")
            else [getattr(pixel_encoder, "embed_dim") * 2**i for i in range(4)]
        )
        config["model"]["pixel_encoder"]["embed_dim"] = getattr(
            pixel_encoder, "embed_dim"
        )
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims
        config["model"]["pixel_encoder"]["depths"] = pixel_encoder.depths
        config["model"]["pixel_encoder"]["cls_token_embed_dims"] = getattr(
            pixel_encoder, "cls_token_embed_dims", pixel_encoder_embed_dims
        )

        pixel_decoder = Decoder(config)

        self.pixel_encoder = pixel_encoder
        self.pixel_decoder = pixel_decoder

        self.slices_encoder_range = list(
            zip([0, *self.pixel_encoder.depths[:-1]], self.pixel_encoder.depths)
        )

        stacking_fn = config["model"]["pixel_encoder"]["stacking_fn"]
        assert (
            stacking_fn in STACKING_FNS
        ), f"Stacking function {stacking_fn} not found in {STACKING_FNS.keys()}"
        self.stacking_fn = STACKING_FNS[stacking_fn]
        self.shape_constraints = config["data"]["augmentations"]["shape_constraints"]
        self.interpolation_mode = "bilinear"

