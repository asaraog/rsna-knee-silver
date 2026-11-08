#!/bin/bash
# idea A: axial PF crops (axial U-Net from skm-ax) -> PF OA, Effusion, Synovitis, 5 folds
cd /workspace; exec > /root/axc.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install --break-system-packages -q pandas pydicom scikit-learn timm segmentation_models_pytorch scipy 2>&1|tail -1
C=train_s16_i384_c130_k60_f-1_cs_lat_gl_cc; mkdir -p /root/cache /workspace/cropsax_rsna /workspace/axc_out
[ -f /root/cache/$C.json ] || CACHE_DIR=/root/cache python3 -u /root/code/build_cache.py --split train --slices 16 --img 384 --crop-mm 130 --fluid -1 --contrast-slots --lat-norm --geo-lat --centroid-crop --workers 32
[ -f /root/cache/$C.json ] || { echo "CACHE FAILED"; exit 1; }
until [ -f /workspace/skm/unet_skm_ax.pt ]; do sleep 60; done
unzip -p /workspace/rsna-knee-abnormality-detection.zip train.csv > /root/code/train.csv
python3 -u /root/code/skm_crops_ax.py /workspace/skm/unet_skm_ax.pt /root/cache/$C /workspace/cropsax_rsna 2>&1 | tee /workspace/axc_out/cropsax.log | grep -E "variant|chosen|DONE"
for f in 0 1 2 3 4; do
  python3 -u /root/code/crop_train.py --crops /workspace/cropsax_rsna --labels /root/code/pubv4_sharp.csv --train-csv /root/code/train.csv --spec "PF OA,Effusion,Synovitis" --tag CROPAX --fold $f --out /workspace/axc_out 2>&1 | tee /workspace/axc_out/axc_f$f.log | grep -aE "^(fold|CROP|Traceback)"
done
ls /workspace/axc_out/wCROPAX_f4.pt && echo "AXC FINISHED"
