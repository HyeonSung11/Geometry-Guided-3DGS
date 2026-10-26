#!/bin/bash

# 데이터를 저장할 폴더 생성
mkdir -p /workspace/data
cd /workspace/data

echo "--- ACID 데이터셋 다운로드 시작 ---"
wget http://schadenfreude.csail.mit.edu:8000/acid.zip
unzip acid.zip && rm acid.zip

echo "--- RE10K 데이터셋 다운로드 시작 ---"
wget http://schadenfreude.csail.mit.edu:8000/re10k.zip
unzip re10k.zip && rm re10k.zip

echo "--- 모든 데이터셋 다운로드 및 압축 해제 완료 ---"