import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import torchvision.utils as vutils
from ....misc.utils import vis_depth_map

# [필수] 'unidepth' 패키지가 (spfsplat) 환경에 설치되어 있어야 합니다.
try:
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Camera, Pinhole, BatchCamera
except ImportError as e:
    print("="*50)
    print(f"[DepthHead 오류] 'unidepth' 패키지를 import할 수 없습니다: {e}")
    print("이 헤드를 사용하려면 'pip install -e .' 등으로 unidepth가 설치되어 있어야 합니다.")
    print("="*50)
    raise e


@dataclass
class DepthHeadCfg:
    """
    DepthHead를 위한 설정 클래스.
    config/model/encoder/head/depth_head.yaml 에서 모델 이름을 지정합니다.
    """
    # lpiccinelli/unidepth-v2-vitl14 등 Hugging Face 모델명
    model_name: str 
    # dataset_re10k.py가 원본 이미지를 0-255 텐서로 주므로 True
    normalize_input: bool = True 


class DepthHead(nn.Module):
    """
    UniDepthV2를 내장하여 수도 뎁스를 생성하는 헤드 모듈.
    SPFSplat의 '토큰'이 아닌 데이터 로더의 '원본 이미지 텐서'를 입력으로 받습니다.
    """
    cfg: DepthHeadCfg

    def __init__(self, net, cfg: DepthHeadCfg):
        super().__init__()
        self.cfg = cfg
        # net (ModelWrapper)에서 device 정보를 가져옵니다.
        self.device = next(net.parameters()).device

        print(f"[DepthHead] Hugging Face에서 UniDepthV2 모델 로드 중: {self.cfg.model_name}")

        # 1. [올바른 방식] from_pretrained로 모델 로드
        # 이 함수가 알아서 config.json과 pytorch_model.bin을 다운로드합니다.
        try:
            self.unidepth_model = UniDepthV2.from_pretrained(self.cfg.model_name)
        except Exception as e:
            print(f"[DepthHead 오류] {self.cfg.model_name} 모델 로드 실패.")
            print("네트워크 연결을 확인하거나, 'unidepth' 패키지가 올바르게 설치되었는지 확인하세요.")
            raise e

        # 2. 모델을 GPU로 이동 및 평가 모드로 설정
        self.unidepth_model = self.unidepth_model.to(self.device)
        self.unidepth_model.eval()

        # 3. [필수] UniDepthV2는 훈련하지 않으므로, 모든 파라미터를 동결(freeze)
        print("[DepthHead] UniDepthV2 모델의 모든 파라미터를 동결합니다.")
        for param in self.unidepth_model.parameters():
            param.requires_grad = False

    @torch.no_grad() # 이 헤드는 학습되지 않으므로, 그래디언트 계산을 중지
    def forward(self, raw_image_tensor: torch.Tensor, camera_tensor: torch.Tensor = None) -> torch.Tensor:
        """
        SPFSplat의 'encoder_tokens'가 아닌, 데이터 로더의
        '원본 이미지 텐서' (e.g., [B, 3, H, W], 0-255 범위)를 입력으로 받습니다.
        """
        
        # 카메라 텐서 처리 (Pinhole 객체로 변환)
        camera_input = None
        original_batch_shape = None # ⬅️ [추가] 바깥에서 선언

        if camera_tensor is not None:
            # (B, 3, 3) 또는 (B, V, 3, 3) 형태의 텐서를 처리
            if camera_tensor.shape[-2:] == (3, 3):
            
                # 텐서가 4D [B, V, 3, 3]인지 확인
                if raw_image_tensor.dim() == 5 and camera_tensor.dim() == 4:
                    original_batch_shape = camera_tensor.shape[:-2]
                    camera_tensor_flat = camera_tensor.reshape(-1, 3, 3)
                    raw_image_tensor = raw_image_tensor.reshape(-1, *raw_image_tensor.shape[-3:]) #raw_image_tensor = torch.Size([1, 3, 256, 256])
                else:
                    # [B, 3, 3] 텐서이므로 평탄화 필요 없음
                    camera_tensor_flat = camera_tensor #camera_tensor_flat.shape = torch.Size([1, 3, 3])

                # ⬇️ [수정] Pinhole/BatchCamera 객체를 직접 만들지 않습니다.
                # 'infer' 함수가 텐서를 바로 처리할 수 있습니다.
                camera_input = Pinhole(K=camera_tensor_flat)
                #cameras = [Pinhole(K=k.unsqueeze(0)) for k in camera_tensor_flat] # ⬅️ 삭제
                #camera_input = BatchCamera.from_camera(cameras).to(self.device) # ⬅️ 삭제
                
                # ⬇️ [대체] 대신, 평탄화된 텐서를 camera_input으로 사용합니다.
                camera_input = camera_tensor_flat.to(self.device)
                
                # ⬇️ [수정] 평탄화 로직을 camera_tensor의 차원 기준으로 이동
                # raw_image_tensor = raw_image_tensor.reshape(-1, *raw_image_tensor.shape[-3:]) # ⬅️ 위로 이동/통합


        #try:
        #    # (B*V, C, H, W) 텐서를 [0, 1] 범위로 정규화하여 이미지로 저장
        #    vutils.save_image(
        #        raw_image_tensor.float() / 255.0, 
        #        "debug_depth_head_input.png", 
        #        nrow=raw_image_tensor.shape[0] # 배치의 모든 이미지를 한 줄로
        #    )
        #except Exception as e:
        #    print(f"[DepthHead 디버깅 오류] 이미지 저장 실패: {e}")


        # UniDepthV2 모델로 추론 실행
        outputs = self.unidepth_model.infer(
            raw_image_tensor, 
            #camera=camera_input, # ⬅️ 이제 camera_input은 [B*V, 3, 3] 텐서이거나 None입니다.
            normalize=self.cfg.normalize_input 
        )

        # 뎁스 맵만 반환 (Z_log와 유사)
        pseudo_depth = outputs['depth'] # [B*V, 1, H, W]

        #try:
        #    # 1. 뎁스 텐서 [B*V, 1, H, W]를 [B*V, H, W]로 만듭니다 (vis_depth_map 입력용)
        #    depth_to_visualize = pseudo_depth.squeeze(1) 
        #    
        #    # 2. 뎁스를 컬러맵 이미지로 변환
        #    # vis_depth_map이 [B*V, H, W]를 받아 [B*V, 3, H, W] 텐서를 반환
        #    colored_depth_maps = vis_depth_map(depth_to_visualize)
        #    # 3. 컬러맵 이미지를 파일로 저장
        #    vutils.save_image(
        #        colored_depth_maps, 
        #        "debug_depth_head_output.png", # ⬅️ 출력 파일
        #        nrow=raw_image_tensor.shape[0] # 배치의 모든 이미지를 한 줄로
        #    )
        #except Exception as e:
        #    print(f"[DepthHead 디버깅 오류] 뎁스 맵 저장 실패: {e}")


        # 만약 원본이 [B, V, ...] 형태였다면, 다시 [B, V, 1, H, W]로 복원
        # ⬇️ [수정] len(original_batch_shape) == 2 대신 original_batch_shape is not None으로 확인
        if camera_input is not None and original_batch_shape is not None: 
            b, v = original_batch_shape
            pseudo_depth = pseudo_depth.reshape(b, v, 1, *pseudo_depth.shape[-2:])

        return pseudo_depth #depth 확인


def create_depth_head(net, cfg):
    return DepthHead(net, cfg)