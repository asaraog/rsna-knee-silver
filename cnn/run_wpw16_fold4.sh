#!/bin/bash
# pw16 on the true-x sagittal order (_po), BLENDED labels (labels_v4fb_sharp.csv = .5 labels_v4_sharp + .5 the Claude whole-report read, sharpened),
# eval on pubv4 like wPWPA (same recipe as pods/pwpa_f4.sh, only --labels differs). fold 4
cd /root/code; exec > /root/pwv4_f4.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65535 2>/dev/null || true
python3 -c "import pandas,pydicom,sklearn,timm" 2>/dev/null || pip install --break-system-packages -q pandas pydicom scikit-learn timm 2>&1|tail -2
C=train_s16_i384_c130_k60_f-1_cs_lat_gl_cc_po
mkdir -p /root/cache /root/out
echo "START $(date)  GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee /root/out/pwv4_f4.status
[ -f /root/cache/$C.json ] || CACHE_DIR=/root/cache python3 -u /root/code/build_cache.py --split train --slices 16 --img 384 --crop-mm 130 --fluid -1 --contrast-slots --lat-norm --geo-lat --centroid-crop --pos-order --workers 32
[ -f /root/cache/$C.json ] || { echo "CACHE FAILED" | tee -a /root/out/pwv4_f4.status; exit 1; }
echo "CACHE READY $(date)" | tee -a /root/out/pwv4_f4.status
cd /root/code && CACHE_DIR=/root/cache WORK=/root/code python3 -u train_cached.py --cache $C --arch convnext_tiny --labels labels_v4fb_sharp.csv --eval-labels pubv4_sharp.csv \
  --no-gold --posw --use 16 --epochs 25 --batch 1 --accum 16 --fold 4 --folds 5 --out /root/out/wPWV4_f4.pt 2>&1 | tee /root/out/pwv4_fold4.log | grep -aE "^(ep |restored|Traceback|saved)"
[ -f /root/out/wPWV4_f4.pt ] && md5sum /root/out/wPWV4_f4.pt > /root/out/wPWV4_f4.md5
echo "PWV4 f4 DONE $(date)" | tee -a /root/out/pwv4_f4.status
# post-train: OOF + gold-58 inference (script rsynced later by the Mac; runs if present)
[ -f /root/code/pw_infer.py ] && cd /root/code && CACHE_DIR=/root/cache WORK=/root/code python3 -u pw_infer.py --cache $C --weights /root/out/wPWV4_f4.pt --fold 4 --tag wPWV4 --out /root/out > /root/out/pw_infer_f4.log 2>&1
echo "ALL f4 DONE $(date)" | tee -a /root/out/pwv4_f4.status
