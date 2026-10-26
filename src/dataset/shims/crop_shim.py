import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8  # ⬅️ [수정] UInt8 임포트 추가
from PIL import Image
from torch import Tensor

from ..types import AnyExample, AnyViews


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    # (이 함수는 기존과 동일: SPFSplat의 [0, 1] 텐서용)
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


# ⬇️ [추가] UniDepthV2의 [0, 255] 원본 텐서용 리사이즈 함수
def rescale_raw(
    image: UInt8[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> UInt8[Tensor, "3 h_out w_out"]:
    h, w = shape
    # 이미지는 [0, 255] uint8 텐서라고 가정
    image_new = rearrange(image, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new)
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    # (이 함수는 기존과 동일)
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    # (이 함수는 기존과 동일)
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    # print(h_in, w_in, h_out, w_out)
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out
    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    # ⬇️ [수정됨] 이 함수가 'raw_image'도 처리하도록 수정
    
    # (A) SPFSplat용 'image' 텐서 처리 (기존 코드)
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape)
    
    # 반환할 딕셔너리 준비
    new_views = {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }

    # (B) [추가] UniDepthV2용 'raw_image' 텐서 처리
    if "raw_image" in views:
        raw_images = views["raw_image"]  # [B, V, 3, H_in, W_in] (uint8)
        *_, h_in, w_in = raw_images.shape
        h_out, w_out = shape
        assert h_out <= h_in and w_out <= w_in

        # 1. 'image'와 동일한 스케일 팩터 계산
        scale_factor = max(h_out / h_in, w_out / w_in)
        h_scaled = round(h_in * scale_factor)
        w_scaled = round(w_in * scale_factor)
        
        # 2. 'raw_image' 리사이즈 (새로운 rescale_raw 함수 사용)
        *batch, c, h, w = raw_images.shape
        raw_images = raw_images.reshape(-1, c, h, w)
        raw_images = torch.stack([rescale_raw(image, (h_scaled, w_scaled)) for image in raw_images])
        raw_images = raw_images.reshape(*batch, c, h_scaled, w_scaled)

        # 3. 'raw_image' 크롭 (이미지만 크롭)
        *_, h_in_scaled, w_in_scaled = raw_images.shape
        row = (h_in_scaled - h_out) // 2
        col = (w_in_scaled - w_out) // 2
        raw_images = raw_images[..., :, row : row + h_out, col : col + w_out]

        # 딕셔너리에 크롭된 raw_image 추가
        new_views["raw_image"] = raw_images

    return new_views


def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    # (이 함수는 기존과 동일)
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }