# RSNA Knee Abnormality Detection 2026: 111th place code

Code for the parts of our 111th-place solution (private 0.947, public 0.960, silver medal) that one teammate built: label scoring, the sagittal CNN pair, the SKM-TEA U-Nets and seven crop readers, and the inference code that runs them. The write-up is in [docs/WRITEUP.md](docs/WRITEUP.md).

## What is here

| Folder | Contents | Write-up section |
|---|---|---|
| `labels/` | `score_labels.py` scores any label file against the 58 expert studies, with bootstrap CIs. `llm_label.py` and `blend_labels.py` are the early local-LLM and regex blend. | 2, Appendix A |
| `cnn/` | `build_cache.py` and `train_cached.py` train the sagittal CNN pair (wCNX, wPW16). `run_wpw16_fold4.sh` is the exact run for one fold. `pw_infer.py` writes out-of-fold predictions. | 3, Appendix D |
| `skm/` | SKM-TEA U-Net training, crop cutting, crop-reader training and prediction. `METHOD.md` describes the method and the evidence for each reader. `runs/` holds the exact run for each crop reader in the final kernel. | 3, Appendix C |
| `inference/` | `ap_lane.py`, the inference code for the CNN pair and the crop readers, exactly as embedded in the final Kaggle kernel. | 1 |
| `figures/` | `make_figs.py` and its data rebuild the three figures byte for byte. | |
| `docs/` | the write-up and its figures. | |

## What is not here

- **Data.** Download the competition data from [Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data). SKM-TEA comes from [Stanford](https://github.com/StanfordMIMI/skm-tea).
- **Label files.** The public table our first readers trained on is [llm_labels_v4_blend](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels). Our other label reads are not published.
- **Weights.** No checkpoints are included.
- **The other models in the final kernel.** The window-MIL stack, ZC-P, ZC-X, MV-ARM, BIG-3D and the referee's crop models were built by the other teammate and are not in this repo. The public half is [R7](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) and [S6](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo), published by their authors.
- **Anything trained on OAI.** The OAI data use agreement does not allow it. Some trainer options mention OAI helper scripts that are not included.

## Reproduce

Each fold trains on one 24 GB GPU (we used RTX 4090 and A6000 pods). Decoding runs on the CPU.

### 1. Score a label file

```bash
python labels/score_labels.py my_labels.csv
```

The file needs `StudyInstanceUID` and the 12 finding columns, in any monotonic scale.

### 2. Sagittal CNN pair

Build the 16-slice cache once. `--pos-order` sorts slices by patient geometry, which the position pooling needs.

```bash
CACHE_DIR=cache python cnn/build_cache.py --split train --slices 16 --img 384 --crop-mm 130 --fluid -1 \
  --contrast-slots --lat-norm --geo-lat --centroid-crop --pos-order --workers 32
```

Train wPW16, one fold at a time (folds 0 to 4):

```bash
CACHE_DIR=cache python cnn/train_cached.py --cache train_s16_i384_c130_k60_f-1_cs_lat_gl_cc_po \
  --arch convnext_tiny --labels labels_v4fb_sharp.csv --eval-labels pubv4_sharp.csv \
  --no-gold --posw --use 16 --epochs 25 --batch 1 --accum 16 --fold 0 --folds 5 --out wPW16_f0.pt
```

wCNX uses the same trainer without `--posw`, on a cache built without `--pos-order`, trained on `pubv4_sharp.csv`. Its original command line was not archived.

Label files ending in `_sharp.csv` hold a rank-percentile probability per finding plus a `<finding>__conf` weight column. The script that sharpened them was not archived either.

### 3. SKM-TEA U-Nets and crop readers

Follow [skm/METHOD.md](skm/METHOD.md). In short:

1. Train the three U-Nets: `skm/skm_train.py` (sagittal), `skm/skm_train_cor.py` and `skm/skm_train_ax.py`.
2. Cut crops with `skm/skm_crops.py`, `skm_crops_v2.py`, `skm_crops_v3.py`, `skm_crops_cor.py` and `skm_crops_ax.py`. `join_crops.py` builds the two-plane input.
3. Train each reader with `skm/crop_train.py`. The exact runs are in `skm/runs/`.

| Reader | Run script |
|---|---|
| crop v1 and its ConvNeXt-small twin | `runs/crop_f0.sh`, `runs/cropb.sh` |
| crop v2 | `runs/cropv2.sh` |
| crop v3 | `runs/cropv3.sh` |
| coronal | `runs/cropc.sh` |
| coronal MCL | `runs/corm.sh` |
| two-plane | `runs/twoplane.sh` |
| axial | `runs/axc.sh` |

The run scripts use pod paths (`/root/...`, `/workspace/...`). Change them to your own.

### 4. Inference

`inference/ap_lane.py` runs inside a Kaggle notebook with the competition data and the weights mounted as datasets. Environment flags (`AP_CROP2`, `AP_CROPC`, `AP_TWOPLANE` and the like) switch each crop reader on.

### 5. Figures

```bash
python figures/make_figs.py
```

`figures/make_data.py` rebuilds `data/label_auc.json` from the label files.

## Citation

Saraogee A, huytngo. 111th Place Solution (18th in the Efficiency Track). Kaggle, 2026. https://doi.org/10.34740/kaggle/w/124605

```bibtex
@misc{saraogee2026rsnaknee,
  author    = {Saraogee, Apoorv and {huytngo}},
  title     = {111th Place Solution (18th in the Efficiency Track)},
  year      = {2026},
  publisher = {Kaggle},
  doi       = {10.34740/kaggle/w/124605},
  url       = {https://doi.org/10.34740/kaggle/w/124605}
}
```

## License

MIT for the code. The data, SKM-TEA and the public label tables keep their own licenses.
