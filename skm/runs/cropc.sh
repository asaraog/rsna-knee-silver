#!/bin/bash
# coronal compartment crops (coronal SKM U-Net) -> classifier 5 folds (Medial/Lateral Meniscus + OA)
cd /workspace; exec > /root/cropc.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install --break-system-packages -q pandas pydicom scikit-learn timm segmentation_models_pytorch scipy 2>&1|tail -1
C=train_s16_i384_c130_k60_f-1_cs_lat_gl_cc
mkdir -p /root/cache /workspace/cropsc_rsna /workspace/cropc_out
[ -f /root/cache/$C.json ] || CACHE_DIR=/root/cache python3 -u /root/code/build_cache.py --split train --slices 16 --img 384 --crop-mm 130 --fluid -1 --contrast-slots --lat-norm --geo-lat --centroid-crop --workers 32
[ -f /root/cache/$C.json ] || { echo "CACHE FAILED"; exit 1; }
unzip -p /workspace/rsna-knee-abnormality-detection.zip train.csv > /root/code/train.csv
python3 -u /root/code/skm_crops_cor.py /workspace/skm/unet_skm_cor.pt /root/cache/$C /workspace/cropsc_rsna 2>&1 | tee /workspace/cropc_out/cropsc.log | grep -E "variant|chosen|DONE"
for f in 0 1 2 3 4; do
  python3 -u /root/code/crop_train.py --crops /workspace/cropsc_rsna --labels /root/code/pubv4_sharp.csv --train-csv /root/code/train.csv --tag CROPC --fold $f --out /workspace/cropc_out 2>&1 | tee /workspace/cropc_out/cropc_f$f.log | grep -aE "^(fold|CROP|Traceback)"
done
ls /workspace/cropc_out/wCROPC_f4.pt && echo "CROPC FINISHED"
