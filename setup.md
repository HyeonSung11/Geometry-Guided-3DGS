# Setup

Tested on Ubuntu 22.04, CUDA 12.1, Python 3.11, PyTorch 2.x. Training was done on one NVIDIA A100 (40 GB); evaluation on RTX 3090.

## 1. Environment

```bash
conda create -n gg3dgs python=3.11 -y
conda activate gg3dgs
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements_uni.txt
```

`requirements_a100.txt` and `requirements_now.txt` are frozen dumps of the environments actually used on the A100 server and the 3090 server; use them only if the main file fails to resolve.

## 2. Pose-aware Gaussian rasterizer

The rasterizer that also back-propagates to the camera pose (`diff_gauss_pose`) is vendored in `diff-gaussian-rasterization/`.

```bash
pip install -e diff-gaussian-rasterization
```

## 3. UniDepthV2

The model code is vendored under `src/unidepth/` for reference, but the runtime import uses the installed `unidepth` package:

```bash
pip install git+https://github.com/lpiccinelli-eth/UniDepth.git
```

The weights `lpiccinelli/unidepth-v2-vitl14` are downloaded from Hugging Face on first run (`src/model/model_wrapper.py`). UniDepthV2 is frozen; it receives no gradients.

## 4. Optional: RoPE CUDA kernels (CroCo v2)

```bash
cd src/model/encoder/backbone/croco/curope/
python setup.py build_ext --inplace
cd ../../../../../..
```

## 5. Pretrained weights

Put these under `pretrained_weights/` (git-ignored):

| File | Purpose | Source |
|---|---|---|
| `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` | Backbone initialization for training | https://download.europe.naverlabs.com/ComputerVision/MASt3R/ |
| `re10k.ckpt` | SPFSplat baseline (for comparison) | https://huggingface.co/RanranHuang/SPFSplat |

## 6. Datasets

Follow [`DATASETS.md`](DATASETS.md) (pixelSplat / SPFSplat chunk format). Expected layout:

```
datasets/
  re10k/
    train/
    test/
  acid/
    train/
    test/
```

Paths are set in `config/dataset/re10k.yaml` and `config/experiment/spfsplat/acid.yaml`.

## 7. Docker (optional)

`Dockerfile` builds a CUDA 12.1 image with Anaconda and the A100 requirements. It was used for the A100 training runs.
