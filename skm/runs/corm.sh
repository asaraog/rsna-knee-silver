#!/bin/bash
# coronal crops + MCL head: Medial Meniscus, Medial OA, MCL | Lateral Meniscus, Lateral OA on the existing coronal crops, 5 folds
cd /workspace; exec > /root/corm.log 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install --break-system-packages -q pandas scikit-learn timm 2>&1|tail -1
mkdir -p /workspace/corm_out; unzip -p /workspace/rsna-knee-abnormality-detection.zip train.csv > /root/code/train.csv
ls /workspace/cropsc_rsna/crops.npy || { echo "CROPS MISSING"; exit 1; }
for f in 0 1 2 3 4; do
  python3 -u /root/code/crop_train.py --crops /workspace/cropsc_rsna --labels /root/code/pubv4_sharp.csv --train-csv /root/code/train.csv --spec "Medial Meniscus,Medial OA,MCL|Lateral Meniscus,Lateral OA" --tag CROPCM --fold $f --out /workspace/corm_out 2>&1 | tee /workspace/corm_out/corm_f$f.log | grep -aE "^(fold|CROP|Traceback)"
done
ls /workspace/corm_out/wCROPCM_f4.pt && echo "CORM FINISHED"
