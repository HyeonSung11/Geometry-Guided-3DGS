from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDepthCfg:
    """ config/loss/depth.yaml에 정의된 설정값 """
    weight: float
    apply_after_step: int
    # ⬇️ [추가] 스케일 정렬(Median Scaling)을 켤지 여부
    use_median_scaling: bool = True 


@dataclass
class LossDepthCfgWrapper:
    """ __init__.py의 LossCfgWrapper가 인식할 래퍼 """
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    """
    SPFSplat 뎁스와 UniDepthV2 수도 뎁스 간의 Loss를 계산합니다.
    (loss_mse.py와 동일한 forward 시그니처를 사용합니다)
    """
    
    def __init__(self, cfg: LossDepthCfgWrapper) -> None:
        super().__init__(cfg)
        # L1 Loss 함수를 내부적으로 사용
        self.loss_fn = nn.L1Loss(reduction="none")

    def forward(
        self,
        # ⬇️ 님의 3단계 계획을 위해 forward 시그니처를 수정합니다.
        # (기존 mse/lpips와 달리 4개의 텐서를 받도록 합니다)
        spfsplat_depth: Float[Tensor, "..."], # (A) SPFSplat 고품질 뎁스
        pseudo_depth: Float[Tensor, "..."],   # (B) UniDepthV2 수도 뎁스
        gaussians: Gaussians, # (기존 시그니처 유지를 위해 받지만 사용 안 함)
        global_step: int,
    ) -> Float[Tensor, ""]:
        
        # 1. 설정된 스텝 이전에는 Loss를 0으로 반환
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=spfsplat_depth.device)

        # 2. 마스크 생성 (유효한 뎁스만 비교)
        with torch.no_grad():
            mask = (pseudo_depth > 1e-3) & (spfsplat_depth > 1e-3)
            if mask.sum() == 0:
                # 비교할 유효한 픽셀이 없으면 0 반환
                return torch.tensor(0, dtype=torch.float32, device=spfsplat_depth.device)

        # 3. 스케일 맞추기 (Median Scaling)
        if self.cfg.use_median_scaling:
            with torch.no_grad():
                median_pseudo = pseudo_depth[mask].median()
                median_spfsplat = spfsplat_depth[mask].median()
                scale_factor = median_pseudo / (median_spfsplat + 1e-8)
            
            # SPFSplat 뎁스의 스케일을 UniDepthV2에 맞춤
            spfsplat_depth_aligned = spfsplat_depth * scale_factor
        else:
            # 스케일 정렬을 사용하지 않음
            spfsplat_depth_aligned = spfsplat_depth

        # 4. L1 Loss 계산 (유효한 마스크 영역만)
        loss = self.loss_fn(
            spfsplat_depth_aligned[mask], 
            pseudo_depth[mask]
        )
        
        return self.cfg.weight * loss.mean()