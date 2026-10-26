# 1. 베이스 이미지 (12.1 환경과 딱 맞는 버전)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 설치 중 지역 설정 등을 묻는 팝업 방지
ENV DEBIAN_FRONTEND=noninteractive

# 2. 필수 시스템 도구 및 SPFSplat용 그래픽/빌드 라이브러리 설치
RUN apt-get update && apt-get install -y \
    git wget bzip2 ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 3. Anaconda 설치 (주석 처리 및 경로 설정 완료)
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /opt/anaconda3 && \
    rm /tmp/anaconda.sh
    
# 시스템이 아나콘다 내부의 파이썬을 먼저 찾도록 설정
ENV PATH=/opt/anaconda3/bin:$PATH

# 4. 작업 경로 설정
WORKDIR /workspace
COPY requirements_a100.txt .

# 5. 패키지 설치 (PyTorch 전용 주소 추가)
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    # [수정] backports.tarfile을 가장 먼저 설치하여 namespace 충돌 방지
    pip install --no-cache-dir backports.tarfile && \
    # 그 다음 기본 패키지 설치
    pip install --no-cache-dir --ignore-installed -r requirements_a100.txt --extra-index-url https://download.pytorch.org/whl/cu121 && \
    # [수정] 나머지 에러 해결 패키지들 설치
    pip install --no-cache-dir \
        "pydantic>=2.0" \
        "psutil" \
        "matplotlib" "pandas" --upgrade \
        "tokenizers>=0.22.0,<=0.23.0" "transformers" "wandb" -U \
        "numpy<2.0" \
        "fvcore" "iopath" && \
    # PyTorch3D 설치
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 6. 데이터 다운로드 스크립트 복사
COPY download_data.sh /workspace/download_data.sh
RUN chmod +x /workspace/download_data.sh

# 기본 셸 실행
CMD ["/bin/bash"]
