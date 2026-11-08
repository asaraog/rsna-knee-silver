#!/bin/bash
# crop-B: second compartment crop model (ConvNeXt-small, seed 1) on the existing v1 crops, 5 folds
cd /workspace; exec > /root/cropb.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install --break-system-packages -q pandas scikit-learn timm 2>&1|tail -1
mkdir -p /workspace/cropb_out; unzip -p /workspace/rsna-knee-abnormality-detection.zip train.csv > /root/code/train.csv
for f in 0 1 2 3 4; do
  python3 -u /root/code/crop_train.py --crops /workspace/crops_rsna --labels /root/code/pubv4_sharp.csv --train-csv /root/code/train.csv --arch convnext_small.fb_in22k_ft_in1k --seed 1 --tag CROPB --fold $f --out /workspace/cropb_out 2>&1 | tee /workspace/cropb_out/cropb_f$f.log | grep -aE "^(fold|CROP|Traceback)"
done
ls /workspace/cropb_out/wCROPB_f4.pt && echo "CROPB FINISHED"
