"""메트릭 깊이맵을 3D 포인트 클라우드로 역투영하고 축별 정규화하는 유틸.

논문 3.2절 / 식 (1):  X_j = d_j * (kappa^v)^{-1} p_j
  - kappa^v : 뷰 v의 카메라 내부 파라미터 행렬 (정규화형 intrinsics, 0~1)
  - p_j     : 픽셀 j의 동차 좌표
  - X_j     : 카메라 좌표계 상의 3D 점

기존 구현은 1채널 메트릭 깊이맵을 그대로 주입했으나, 본 모듈은 논문 서술대로
깊이맵을 3D 포인트맵(xyz)으로 역투영한 뒤 각 축을 정규화하여 반환한다.
"""
import torch


def unproject_depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """깊이맵 -> 카메라 좌표계 3D 포인트맵.

    Args:
        depth: [B, V, 1, H, W] 또는 [B, V, H, W]  (메트릭 깊이)
        intrinsics: [B, V, 3, 3]  (정규화형: fx,fy,cx,cy 가 0~1 범위)
    Returns:
        points: [B, V, 3, H, W]  (x, y, z = 카메라 좌표)
    """
    if depth.dim() == 5:
        z = depth[:, :, 0]  # [B, V, H, W]
    else:
        z = depth
    B, V, H, W = z.shape

    # 정규화형 intrinsics 를 픽셀 단위로 환원 (loss_reproj 의 unnorm 규약과 동일)
    fx = (intrinsics[..., 0, 0] * W)[..., None, None]  # [B, V, 1, 1]
    fy = (intrinsics[..., 1, 1] * H)[..., None, None]
    cx = (intrinsics[..., 0, 2] * W)[..., None, None]
    cy = (intrinsics[..., 1, 2] * H)[..., None, None]

    vs = torch.arange(H, device=depth.device, dtype=z.dtype)
    us = torch.arange(W, device=depth.device, dtype=z.dtype)
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")  # [H, W]
    grid_u = grid_u[None, None]  # [1, 1, H, W]
    grid_v = grid_v[None, None]

    x = (grid_u - cx) * z / (fx + eps)
    y = (grid_v - cy) * z / (fy + eps)
    points = torch.stack([x, y, z], dim=2)  # [B, V, 3, H, W]
    return points


def normalize_pointmap_per_axis(points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """포인트맵의 각 축(x, y, z)을 뷰별로 표준화(mean-std)한다.

    논문 3.2절 "포인트 맵의 각 축을 정규화" 에 해당.
    Args:
        points: [B, V, 3, H, W]
    Returns:
        normalized points: [B, V, 3, H, W]
    """
    mean = points.mean(dim=(-1, -2), keepdim=True)  # [B, V, 3, 1, 1]
    std = points.std(dim=(-1, -2), keepdim=True)
    return (points - mean) / (std + eps)


def unproject_and_normalize(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """깊이맵 -> 역투영(식 1) -> 축별 정규화 까지 한 번에 수행.

    Returns:
        [B, V, 3, H, W] 정규화된 3D 포인트맵 (주입용).
    """
    points = unproject_depth_to_points(depth, intrinsics)
    points = normalize_pointmap_per_axis(points)
    return points
