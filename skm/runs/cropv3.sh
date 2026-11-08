#!/bin/bash
# crop v3: medial-edge crop (MCL, Baker's) + bone crop over both compartments (Contusion) -> classifier 5 folds; then pull all_slices.csv sides
cd /workspace; exec > /root/cropv3.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install --break-system-packages -q pandas pydicom scikit-learn timm segmentation_models_pytorch scipy 2>&1|tail -1
C=train_s16_i384_c130_k60_f-1_cs_lat_gl_cc
mkdir -p /root/cache /workspace/crops3_rsna /workspace/crop3_out
[ -f /root/cache/$C.json ] || CACHE_DIR=/root/cache python3 -u /root/code/build_cache.py --split train --slices 16 --img 384 --crop-mm 130 --fluid -1 --contrast-slots --lat-norm --geo-lat --centroid-crop --workers 32
[ -f /root/cache/$C.json ] || { echo "CACHE FAILED"; exit 1; }
unzip -p /workspace/rsna-knee-abnormality-detection.zip train.csv > /root/code/train.csv
python3 -u /root/code/skm_crops_v3.py /workspace/skm/unet_skm.pt /root/cache/$C /workspace/crops3_rsna 2>&1 | tee /workspace/crop3_out/crops3.log | tail -2
for f in 0 1 2 3 4; do
  python3 -u /root/code/crop_train.py --crops /workspace/crops3_rsna --labels /root/code/pubv4_sharp.csv --train-csv /root/code/train.csv --spec "MCL,Baker's|Contusion" --tag CROP3 --fold $f --out /workspace/crop3_out 2>&1 | tee /workspace/crop3_out/crop3_f$f.log | grep -aE "^(fold|CROP|Traceback)"
done
ls /workspace/crop3_out/wCROP3_f4.pt && echo "CROPV3 FINISHED"
