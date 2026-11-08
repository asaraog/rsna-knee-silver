#!/bin/bash
# SKM-TEA seg-guided compartment crops -> crop classifier fold 0 (+ wLBC_f0 on the same val fold as the per-label baseline)
cd /workspace; exec > /root/crop_f0.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install --break-system-packages -q pandas pydicom scikit-learn timm segmentation_models_pytorch scipy 2>&1|tail -1
C=train_s16_i384_c130_k60_f-1_cs_lat_gl_cc
mkdir -p /root/cache /root/out /workspace/crop_out /workspace/crops_rsna
[ -f /root/cache/$C.json ] || CACHE_DIR=/root/cache python3 -u /root/code/build_cache.py --split train --slices 16 --img 384 --crop-mm 130 --fluid -1 --contrast-slots --lat-norm --geo-lat --centroid-crop --workers 32
[ -f /root/cache/$C.json ] || { echo "CACHE FAILED"; exit 1; }
unzip -p /workspace/rsna-knee-abnormality-detection.zip train.csv > /root/code/train.csv
cp /workspace/skm/*.py /workspace/lb_out/wLBC_f0.pt /root/code/
python3 -u /root/code/skm_crops.py /workspace/skm/unet_skm.pt /root/cache/$C /workspace/crops_rsna 2>&1 | tee /workspace/crop_out/crops.log | tail -3
python3 -u /root/code/crop_train.py --crops /workspace/crops_rsna --labels /root/code/pubv4_sharp.csv --train-csv /root/code/train.csv --fold 0 --out /workspace/crop_out 2>&1 | tee /workspace/crop_out/crop_f0.log | grep -aE "^(ep |fold|CROP|Traceback)"
cd /root/code && CACHE_DIR=/root/cache python3 score_moaks.py predict --cache $C --tc-dir /root/code --ckpt /root/code/wLBC_f0.pt --use 8 --batch 2 --lenient --out /workspace/crop_out/lbc_f0_rsna.csv 2>&1 | tail -2
[ -s /workspace/crop_out/wCROP_f0.pt ] && echo "CROP f0 FINISHED"
