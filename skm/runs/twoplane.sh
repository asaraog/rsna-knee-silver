#!/bin/bash
# lever #1: two-plane compartment model (8 sagittal + 8 coronal crops per compartment), 5 folds
cd /workspace; exec > /root/twoplane.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install --break-system-packages -q pandas scikit-learn timm 2>&1|tail -1
mkdir -p /workspace/tp_out; unzip -p /workspace/rsna-knee-abnormality-detection.zip train.csv > /root/code/train.csv
[ -f /root/tp/crops.npy ] || python3 -u /root/code/join_crops.py /workspace/crops_rsna /workspace/cropsc_rsna /root/tp 2>&1 | tail -2
for f in 0 1 2 3 4; do
  python3 -u /root/code/crop_train.py --crops /root/tp --labels /root/code/pubv4_sharp.csv --train-csv /root/code/train.csv --tag CROPSC --fold $f --batch 6 --out /workspace/tp_out 2>&1 | tee /workspace/tp_out/tp_f$f.log | grep -aE "^(fold|CROP|Traceback)"
done
ls /workspace/tp_out/wCROPSC_f4.pt && echo "TWOPLANE FINISHED"
