from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange
from torch import nn

from .croco.blocks import DecoderBlock
from .croco.croco import CroCoNet
from .croco.misc import fill_default_args, freeze_all_params, transpose_to_landscape, is_symmetrized, interleave, \
    make_batch_symmetric
from .croco.patch_embed import get_patch_embed
from .backbone import Backbone
from ....geometry.camera_emb import get_intrinsic_embedding, get_intrinsic_positional_embedding

inf = float('inf')


croco_params = {
    'ViTLarge_BaseDecoder': {
        'enc_depth': 24,
        'dec_depth': 12,
        'enc_embed_dim': 1024,
        'dec_embed_dim': 768,
        'enc_num_heads': 16,
        'dec_num_heads': 12,
        'pos_embed': 'RoPE100',
        'img_size': (512, 512),
    },
}

default_dust3r_params = {
    'enc_depth': 24,
    'dec_depth': 12,
    'enc_embed_dim': 1024,
    'dec_embed_dim': 768,
    'enc_num_heads': 16,
    'dec_num_heads': 12,
    'pos_embed': 'RoPE100',
    'patch_embed_cls': 'PatchEmbedDust3R',
    'img_size': (512, 512),
    'head_type': 'dpt',
    'output_mode': 'pts3d',
    'depth_mode': ('exp', -inf, inf),
    'conf_mode': ('exp', 1, inf)
}

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


@dataclass
class BackboneCrocoCfg:
    name: Literal["croco", "croco_multi"]
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token"] = 'token'  # linear or dpt


class AsymmetricCroCo(CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        self.cfg = cfg
        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_encoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
        elif self.intrinsics_embed_loc == 'decoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_decoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3

        self.patch_embed_cls = cfg.patch_embed_cls
        self.croco_args = fill_default_args(croco_params[cfg.model], CroCoNet.__init__)

        super().__init__(**croco_params[cfg.model])

        if cfg.asymmetry_decoder:
            self.dec_blocks2 = deepcopy(self.dec_blocks)  # This is used in DUSt3R and MASt3R

        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'pe_and_token':
            self.intrinsic_encoder = nn.Linear(9, 1024)

        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'ray_token':
            self.intrinsic_encoder = nn.Linear(768, 1024)

        self.zero_convs = []
        for i in range(len(self.dec_blocks_pc) + 1):
            self.zero_convs.append(self.make_zero_conv(self.dec_embed_dim).cuda())
        self.zero_convs = nn.ModuleList(self.zero_convs)   
        self.intrinsic_adapter = nn.Linear(1024, 768)

       
        # self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        # self.set_freeze(freeze)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans)
        # [논문 일치] 1채널 깊이맵이 아니라 3채널 3D 포인트맵(xyz)을 입력으로 받는다.
        self.patch_embed_point_cloud = get_patch_embed(self.patch_embed_cls, img_size, patch_size, self.dec_embed_dim, in_chans=3)


    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ['none', 'mask', 'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_decoder':  [self.mask_token, self.patch_embed, self.enc_blocks, self.enc_norm, self.decoder_embed, self.dec_blocks, self.dec_blocks2, self.dec_norm],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def _encode_image(self, image, true_shape, intrinsics_embed=None, intrinsics_pos_embed=None, pose_embedding=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape) # (bv, 256, 1024), (bv, 256, 2)
       
        P = x.shape[1]
        if intrinsics_pos_embed is not None:
            x = x + intrinsics_pos_embed

        if intrinsics_embed is not None:

            if self.intrinsics_embed_type == 'linear':
                x = x + intrinsics_embed
            elif self.intrinsics_embed_type == 'token':
                x = torch.cat((x, intrinsics_embed), dim=1)
                add_pose = pos[:, 0:1, :].clone()
                add_pose[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
                pos = torch.cat((pos, add_pose), dim=1)
          

        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, P 

    def make_zero_conv(self, channels):
        return nn.Sequential(zero_module(conv_nd(1, channels, channels, 1, padding=0)))

    def _decoder(self, feat, pose, point_cloud, point_cloud_pos, extra_embed=None):
        b, v, l, c = feat.shape
        final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)

        pc = None
        pc_pos = None
        if point_cloud is not None:
            # 뎁스 피처와 위치를 (b*v, ...)로 rearrange
            pc = rearrange(point_cloud, "b v l c -> (b v) l c")
            pc_pos = rearrange(point_cloud_pos, "b v l c -> (b v) l c")
            
            # 뎁스 피처를 zero_convs[0]으로 변환하여 f에 더함
            pc_feat_add = self.zero_convs[0](pc.transpose(-1, -2)).transpose(-1, -2)
            f = f + pc_feat_add

        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v)
        final_output.append(f)

        def generate_ctx_views(x):
            b, v, l, c = x.shape
            ctx_views = x.unsqueeze(1).expand(b, v, v, l, c)
            mask = torch.arange(v).unsqueeze(0) != torch.arange(v).unsqueeze(1)
            ctx_views = ctx_views[:, mask].reshape(b, v, v - 1, l, c)  # B, V, V-1, L, C
            ctx_views = ctx_views.flatten(2, 3)  # B, V, (V-1)*L, C
            return ctx_views.contiguous()

        pos_ctx = generate_ctx_views(pose)
        # enumerate를 사용하여 루프 인덱스(i) 추적
        for i, (blk1, blk2) in enumerate(zip(self.dec_blocks, self.dec_blocks2)):
            feat_current = final_output[-1]
            feat_current_ctx = generate_ctx_views(feat_current)

            # --- 5a. [신규] 뎁스 디코더 블록 (self-attention) ---
            if pc is not None and i < len(self.dec_blocks_pc):
                # 뎁스 피처(pc)를 자체 디코더 블록으로 업데이트
                pc = self.dec_blocks_pc[i](pc, pc_pos) 
            
            # --- 5b. (기존) SPFSplat 비대칭 블록 ---
            # img1 side (View 0)
            f1, _ = blk1(feat_current[:, 0].contiguous(), feat_current_ctx[:, 0].contiguous(), pose[:, 0].contiguous(), pos_ctx[:, 0].contiguous())
            f1 = f1.unsqueeze(1) # (b, 1, l, c)
            
            # img2 side (View 1 ~ V-1)
            f2, _ = blk2(rearrange(feat_current[:, 1:], "b v l c -> (b v) l c"),
                         rearrange(feat_current_ctx[:, 1:], "b v l c -> (b v) l c"),
                         rearrange(pose[:, 1:].contiguous(), "b v l c -> (b v) l c"),
                         rearrange(pos_ctx[:, 1:].contiguous(), "b v l c -> (b v) l c"))
            f2 = rearrange(f2, "(b v) l c -> b v l c", b=b, v=v-1) # (b, v-1, l, c)
            
            # f1과 f2를 다시 합침
            f_out = torch.cat((f1, f2), dim=1) # (b, v, l, c)

            # --- 5c. [신규] 업데이트된 뎁스 피처 퓨전 ---
            if pc is not None and i < len(self.dec_blocks_pc):
                # zero_convs[i+1]로 뎁스 피처 변환
                pc_feat_add = self.zero_convs[i+1](pc.transpose(-1, -2)).transpose(-1, -2)
                # (b*v, l, c) -> (b, v, l, c)로 변환
                pc_feat_add = rearrange(pc_feat_add, "(b v) l c -> b v l c", b=b, v=v)
                
                # 이미지 피처에 더함
                f_out = f_out + pc_feat_add

            # store the result
            final_output.append(f_out)

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        last_feat = rearrange(final_output[-1], "b v l c -> (b v) l c")
        last_feat = self.dec_norm(last_feat)
        final_output[-1] = rearrange(last_feat, "(b v) l c -> b v l c", b=b, v=v)
        return final_output

    def forward(self,
                context: dict,
                target_num_views: int = 0,
                symmetrize_batch=False,
                return_views=False,
                all_uni_depths = None
                ):
        b, v, _, h, w = context["image"].shape
            
        images_all = context["image"]

        # camera embedding in the encoder
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            intrinsic_embedding = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree) # (b, v, 25, h w)
            images_all = torch.cat((images_all, intrinsic_embedding), dim=2) # (b, v, 28, h w)

        
        intrinsic_embedding_all = None
        if self.intrinsics_embed_loc == 'encoder' and (self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'linear'):
            intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2))
            intrinsic_embedding_all = rearrange(intrinsic_embedding, "b v c -> (b v) c").unsqueeze(1)

        

        # step 1: encoder input images
        images_all = rearrange(images_all, "b v c h w -> (b v) c h w")
        shape_all = torch.tensor(images_all.shape[-2:])[None].repeat(b*v, 1)


        feat, pos, P = self._encode_image(images_all, shape_all, intrinsic_embedding_all)

        feat = rearrange(feat, "(b v) l c -> b v l c", b=b, v=v)
        pos = rearrange(pos, "(b v) l c -> b v l c", b=b, v=v)

        # --- [신규] 뎁스맵 인코딩 (Step 1.5) ---
        point_cloud, point_cloud_pos = None, None
        if all_uni_depths is not None:
            # all_uni_depth.shape: (b, v, h, w)
            
            # 1. 채널 차원 추가: (b, v, 1, h, w)
            point_cloud_input = all_uni_depths
            
            # 2. 이미지와 동일하게 rearrange: (b*v, 1, h, w)
            point_cloud_input = rearrange(point_cloud_input, "b v c h w -> (b v) c h w")

            # 3. 뎁스맵 패치 임베딩 (shape_all은 Step 1의 (b*v, 2) 텐서 재사용)
            point_cloud, point_cloud_pos = self.patch_embed_point_cloud(
                point_cloud_input, 
                true_shape=shape_all
            )
            # point_cloud.shape: (b*v, l_pc, c_pc)
            
            # 4. (b, v, ...)로 다시 rearrange
            point_cloud = rearrange(point_cloud, "(b v) l c -> b v l c", b=b, v=v) #torch.Size([1, 3, 256, 768])
            point_cloud_pos = rearrange(point_cloud_pos, "(b v) l c -> b v l c", b=b, v=v) #torch.Size([1, 3, 256, 2])

            if intrinsic_embedding_all is not None:
                # 1. Intrinsic Token을 Depth Feature 차원(768)로 변환
                # intrinsic_embedding_all은 (B*V, 1, 1024) 형태입니다.
                intrinsic_pc_adapter = self.intrinsic_adapter(intrinsic_embedding_all)
                
                # B, V, L, C 형태로 reshape (B, V, 1, 768)
                intrinsic_pc_adapter = rearrange(intrinsic_pc_adapter, "(b v) l c -> b v l c", b=b, v=v)
                
                # 2. ★ Depth Token 시퀀스의 맨 뒤에 추가 (dim=2는 시퀀스 길이 L 차원)
                point_cloud = torch.cat([point_cloud, intrinsic_pc_adapter], dim=2)
                
                # 3. Positional Embedding도 동일하게 맨 뒤에 0으로 채운 위치 정보를 추가
                #    (위치 정보는 2차원(2)이므로, 768차원 어댑터 크기와 맞춰 생성)
                pad_pos = torch.zeros_like(point_cloud_pos[:, :, 0:1, :]) # (B, V, 1, 2)
                point_cloud_pos = torch.cat([point_cloud_pos, pad_pos], dim=2)



        # ----------------------------------------
        
        # Context 뷰 개수 계산 
        v_cxt = v - target_num_views

        # step 2: decoder without target views
        dec_feat = self._decoder(feat[:,:v-target_num_views], pos[:,:v-target_num_views].contiguous(), point_cloud[:, :v_cxt].contiguous() if point_cloud is not None else None, point_cloud_pos[:, :v_cxt].contiguous() if point_cloud_pos is not None else None)

        # intrinsics 토큰 제거
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token':
            dec_feat = list(dec_feat)
            for i in range(len(dec_feat)):
                dec_feat[i] = dec_feat[i][:, :, :P]
               

         # step 3: decoder with target views
        if target_num_views != 0:
            dec_feat_w_tgt = self._decoder(feat, pos, point_cloud.contiguous() if point_cloud is not None else None, point_cloud_pos.contiguous() if point_cloud_pos is not None else None) # list of (b, v, hw+1 or hw, c)
            if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token':
                dec_feat_w_tgt = list(dec_feat_w_tgt)
                for i in range(len(dec_feat_w_tgt)):
                    dec_feat_w_tgt[i] = dec_feat_w_tgt[i][:, :, :P]
                    
        else:
            dec_feat_w_tgt = None

        shape = rearrange(shape_all, "(b v) c -> b v c", b=b, v=v)
        images = rearrange(images_all, "(b v) c h w -> b v c h w", b=b, v=v)


        out = dict()
        out['dec_feat'] = dec_feat
        out['shape'] = shape
        out['images'] = images

        if dec_feat_w_tgt is not None:
            out['dec_feat_w_tgt'] = dec_feat_w_tgt
        
        return out

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024
