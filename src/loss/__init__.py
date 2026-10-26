from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_reproj import LossReproj, LossReprojCfgWrapper
from .loss_depth import LossDepth, LossDepthCfgWrapper



LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossReprojCfgWrapper: LossReproj,
    LossDepthCfgWrapper: LossDepth,

}

LossCfgWrapper =  LossLpipsCfgWrapper | LossMseCfgWrapper  | LossReprojCfgWrapper | LossDepthCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
