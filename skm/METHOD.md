# Segmentation-guided crop readers

A U-Net finds the anatomy in each knee, then small classifiers read fixed windows cut around that anatomy. Each finding gets a view
of the region a radiologist would look at, at full resolution, instead of the whole 130 mm field. The readers join the final blend
as rank members, each on its own findings.

This is not template registration. Warping every knee into a template space and training the usual model gave no gain. The crops
keep each knee in its own space and only move the window.

## 1. Segmentation: SKM-TEA U-Nets

- Data: [SKM-TEA](https://github.com/StanfordMIMI/skm-tea) (Stanford, 155 DESS knees) with six labels: patellar cartilage, femoral
  cartilage, medial and lateral tibial cartilage, medial and lateral meniscus. Medial vs lateral is a label, so the network learns
  the side from anatomy.
- Sagittal U-Net (`skm_train.py`): smp U-Net, ResNet-34 encoder, 2D, 384 px at 0.417 mm/px. Contrast augmentation mixes the two
  DESS echoes plus gamma and noise, so it transfers to fat-suppressed PD/T2 sagittals. Held-out Dice 0.845.
- Coronal U-Net (`skm_train_cor.py`): the same volumes resliced to coronal, with random left-right flips so medial vs lateral comes
  from anatomy rather than image side. Dice 0.806.
- Axial U-Net (`skm_train_ax.py`): the same volumes resliced to axial. Dice 0.819.
- Transfer check without labels: on RSNA, the per-slice medial/lateral call is monotone in slice position in 99.7% of 4,407
  studies, and its sign matches the DICOM laterality tag in 98.8% of tagged studies.
- The U-Nets are exported as TorchScript fp16 (`export_weights.py`), so inference needs no segmentation library.

## 2. Crops

Crops are cut from the same 16-slice, 384 px, 130 mm stack the sagittal CNN pair reads, so inference reuses the decoded slices.
Per crop: the 8 slices with the most target-label area, a square window on that slice's label centroid, resized to 224 px.

| Reader | Plane | Window | Findings | Script | Weight |
|---|---|---|---|---|---:|
| crop v1 | sagittal | 56 mm on each compartment | Medial/Lateral Meniscus, Medial/Lateral OA | `skm_crops.py` | 0.3 |
| crop v2 | sagittal | 64 mm notch; 64 mm patellofemoral | ACL; PF OA | `skm_crops_v2.py` | 0.5 |
| crop v3 | sagittal | 80 mm, medial edge, shifted posterior | Baker's | `skm_crops_v3.py` | 0.3 |
| coronal | coronal | 56 mm on each compartment | Medial/Lateral Meniscus, Medial/Lateral OA | `skm_crops_cor.py` | 0.5 |
| coronal MCL | coronal | the coronal crops, MCL head | MCL | `skm_crops_cor.py` | 0.3 |
| two-plane | both | 8 sagittal + 8 coronal crops per compartment | Medial/Lateral Meniscus, Medial/Lateral OA | `join_crops.py` | 0.2 |
| axial | axial | 8 × 80 mm on the patellar cartilage | PF OA, Effusion, Synovitis | `skm_crops_ax.py` | 0.3 |

## 3. Readers

`crop_train.py`: ConvNeXt-Tiny (ImageNet-22k), each of the 8 crops through the backbone, mean + max over slices, linear head.
Trained on the public LLM label table, 5 folds split by a hash of the report text (the same rule as `cnn/train_cached.py`), about
15 minutes per fold on an RTX 4090. The loss covers only the findings a crop can see. The exact run for each reader is in `runs/`.

## 4. Blending

`crop_predict.py` writes one CSV per reader in rank units. The final kernel blends each reader after the referees:
`col = (1 - w) * rank(col) + w * rank(reader)`, with the weights in the table above.

## 5. Evidence

Each reader was added alone, read on the 58 expert studies, then submitted.

| Step | Expert set | Other check | Public board |
|---|---:|---|---:|
| + crop v1 | +0.0051 | OAI MOAKS, compartments +0.0034 | 0.954 |
| + crop v2 | +0.0024 | OAI MOAKS +0.0028 | 0.955 |
| + crop v3, crop v2 at 0.5 | +0.0007 | OAI MOAKS +0.0020 | 0.956 |
| + coronal | +0.0037 | RSNA out-of-fold, compartments +0.0115 | 0.957 |
| + two-plane | +0.0015 | RSNA out-of-fold, compartments +0.0033 | 0.958 |

OAI MOAKS cannot judge the coronal readers, because OAI's coronal series differ; those were judged on RSNA out-of-fold and the expert set.

## 6. What did not work

| Idea | Result |
|---|---|
| warping knees into an atlas space | no gain out-of-fold |
| pose-normalised crops | lower on every finding out-of-fold |
| suprapatellar pouch crops for Synovitis | below the axial crop on the expert set |
| ConvNeXt-small twins of the readers | same crops, same errors |
| coronal joint-space width for OA | AUC 0.64 medial, 0.50 lateral |
| coronal bone crops for Contusion | lower on the board |
| Hoffa-pad crops for Synovitis | +0.004 on the expert set, -0.001 on the board |

## Weights

No weights are published.
