# 111th Place Solution (18th in the Efficiency Track)

## Crop Readers Placed by U-Nets Trained on SKM-TEA, a Public Knee Segmentation Set

RSNA Knee Abnormality Detection 2026, team huytngo and apoorvsara. Private **0.947**, public 0.960, rank 111 of 6,299, silver medal. [Efficiency Track](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb): **18th**.

**Abstract**

Only 58 of the 4,407 training studies carry expert labels. Every training study has a radiology report, and the test set has none. Like most teams, we read the reports into soft labels and trained image models on them.

Our final submission blends two halves per finding. The hybrid half fuses an eight-arm window-MIL stack with two sagittal ConvNeXt readers, then adds a medial/lateral referee and twelve specialist members, most of them reading crops placed by SKM-TEA U-Nets. The public half is two public models, heliosli's [R7](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) and nartaa's [S6](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo). Alone, the halves score 0.944 and 0.941 on the private set. Together they score **0.947**.

Our labels set the ceiling. They came from the reports alone and score **0.909** macro AUC on the 58 expert studies. The 1st-place labels, read from the images as well as the reports, score **0.943** on the same studies.

Thanks to RSNA, the annotating radiologists and Kaggle. Thanks also to @nartaa, @heliosli and @dreaddevelopment, whose public models and recipes carried much of our score.

## 1. Overview

Every part's predictions are turned into ranks before they are blended, so no part needs calibrating against another.

![Figure 1. Solution overview](fig/fig1_overview.png)

| Part | Models | Weight |
|---|---|---:|
| Window-MIL stack | 8 arms: CoAtNet-2, ConvNeXt-V2-B, EfficientNet-B0 | 0.7 of the fuse |
| Sagittal CNN pair | ConvNeXt-Tiny, 2 × 5 folds | 0.3 of the fuse |
| Referee | medial/lateral and patellofemoral corrections | λ 0.1 and 0.2 |
| Specialists | 12 members, each on its own findings | 0.2 to 0.5 per finding |
| Public half | [R7](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) and [S6](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo), as published | 0.5 each |
| Final | hybrid half and public half, per finding | 0.5 each |

Synovitis and PF OA take only 0.2 and 0.4 from the public half, where the hybrid half was stronger on the expert set. The stack and the CNN pair run in parallel, one per T4, and the public stages follow behind their own deadlines. The kernel finishes about ten minutes inside the 9 h limit. Full model parameters are in the appendix.

## 2. Labels

Our targets average two LLM reads of every report, a consensus read and a Claude read, for **0.909** macro AUC on the expert set (Appendix A). The reports span nine languages, and the signal sits in polarity, "intact" against "tear", not in whether a finding is named. Better labels converted to better models almost one for one: a teacher 0.046 better on the expert set gave a student 0.039 better.

After the close, three teams published their labels, and we scored them on the same 58 studies.

| Label set | How it was made | Macro AUC | Synovitis | Effusion |
|---|---|---:|---:|---:|
| Ours | reports only, LLM + Claude | 0.909 | 0.802 | 0.866 |
| [Public frontier blend](https://www.kaggle.com/datasets/cdeotte/rsna-knee-frontier-llm-hybrid-labels) | three frontier LLMs | 0.927 | 0.852 | 0.935 |
| [1st place](https://www.kaggle.com/datasets/tascj0/rsna-2026-knee-agent-labels) | GPT Astra reading report and images | 0.943 | 0.881 | 0.945 |

![Figure 2. Training-label sets on the 58 expert studies](fig/fig2_labels.png)

Image-read labels lead most where reports are least reliable: Synovitis, Effusion and Contusion. The 1st-place guide was tuned on these 58 studies, so part of its lead is in-sample.

## 3. Models

Every model's parameters are in the appendix. We set the weights one factor at a time: each moved only on a local gain, and stayed only if the public board held.

- **Window-MIL stack, 0.7 of the fuse.** Eight arms on Dread Development's public [Raptor](https://www.kaggle.com/code/dreaddevelopment/the-original-raptor-12-findings-full-release) recipe, with CoAtNet-2 [1], ConvNeXt-V2-B [2] and EfficientNet-B0 [3] backbones. Each study becomes 42 windows from a 64-slice stack, and each arm pools its window features by gated attention into one score per finding (Appendix B).
- **Sagittal CNN pair, 0.3 of the fuse.** Two ConvNeXt-Tiny [4] readers over 16 sagittal slices, one with top-k pooling and one with a learned medial-to-lateral pooling. The second only worked once slices were sorted by patient geometry: InstanceNumber ran one way in 295 of 633 sagittal series and the other way in 338.
- **Specialists.** Three U-Nets [5] trained on [SKM-TEA](https://github.com/StanfordMIMI/skm-tea) [6] place crops on the compartments, the notch, the patellofemoral joint and the medial edge. Seven ConvNeXt-Tiny crop readers took public from 0.954 to 0.958 (Appendix C). Five more add attention-MIL [7] crops, a multi-view transformer, a 3D CNN [8] and a reader trained on OAI knees with MOAKS [9] grades (Appendix D).
- **Referee.** It corrects medial versus lateral for OA and menisci from the U-Net side calls and two crop models (λ 0.1), and PF OA from an anterior crop (λ 0.2).
- **Public half.** [R7](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) and [S6](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo) ran as published. On the private set, R7 adds about +0.002 beside S6, and S6 about +0.001 beside R7.

## 4. Testing

![Figure 3. How the solution developed](fig/fig3_history.png)

| Check | What it is | How we read it |
|---|---|---|
| Expert set | the 58 RSNA studies with expert labels | AUC per finding |
| OAI replica | 1,497 OAI knees run through the full kernel, 1,244 with expert MOAKS grades | paired bootstrap by participant |
| RSNA holdout | 217 RSNA studies kept out of training, report labels | paired bootstrap, 2,000 draws |
| Public board | Kaggle's public split | the final call |

Private ran about **0.013** below public throughout, and the order held (Spearman **0.95**), so the public board was an honest gate.

## 5. Efficiency Track

Our second final pick, LITE V29, placed 18th on the [Efficiency Track leaderboard](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb). It is a reduced window-MIL stack that runs in about 50 minutes and scores 0.939 private, 0.950 public.

## 6. Conclusion

The two halves complement each other. On the private set the hybrid half alone scores 0.944 and the public half 0.941, and the blend reaches 0.947.

Variants on the final blend landed between 0.944 and 0.947, and a few ideas lost outright (Appendix E). What we would do differently:

1. **Read the images when making labels.** [1st place](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752468) labeled with GPT Astra reading the scans as well as the reports. In their [matched comparison](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752474), those labels alone lifted the same CoAtNet from 0.922 to 0.935 public. Ours came from the reports alone.
2. **Build one strong student before blending.** [7th place](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752217) won gold with a single 2.5D network at 0.958 private. Our hybrid half has fourteen parts and scores 0.944 alone.
3. **Put anatomy into the main model.** [1st place](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752474) trained pooling and attention on anatomical and lesion boxes, and [5th place](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752190) cut its main readers' crops from a bone segmentation. Our SKM-TEA crops arrived late, as add-ons.
4. **Validate on more than 58 studies.** [1st place](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752474) selected models by a probability-weighted AUC on its continuous labels, and [6th place](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/753159) checked every label release against its own past board scores.
5. **Sort slices by patient geometry, never by InstanceNumber.**

## Code

Training and inference code for the sagittal CNN pair, the SKM-TEA U-Nets and the seven crop readers, with the exact run for each model: [github.com/asaraog/rsna-knee-silver](https://github.com/asaraog/rsna-knee-silver).

## Credits

Team [huytngo](https://www.kaggle.com/huytngo) and [apoorvsara](https://www.kaggle.com/apoorvsara). [@nartaa](https://www.kaggle.com/nartaa) for [S6](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo), [@heliosli](https://www.kaggle.com/heliosli) for [R7](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss), [@dreaddevelopment](https://www.kaggle.com/dreaddevelopment) for the Raptor recipe and its CC0 labels. The [SKM-TEA](https://github.com/StanfordMIMI/skm-tea) team at Stanford for the segmentations, and the [OAI](https://nda.nih.gov/oai) for the MOAKS grades.

## References

1. Dai Z, Liu H, Le QV, Tan M. CoAtNet: Marrying Convolution and Attention for All Data Sizes. NeurIPS, 2021. [arXiv:2106.04803](https://arxiv.org/abs/2106.04803)
2. Woo S, et al. ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders. CVPR, 2023. [arXiv:2301.00808](https://arxiv.org/abs/2301.00808)
3. Tan M, Le QV. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
4. Liu Z, et al. A ConvNet for the 2020s. CVPR, 2022. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
5. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
6. Desai AD, et al. SKM-TEA: A Dataset for Accelerated MRI Reconstruction with Dense Image Labels for Quantitative Clinical Evaluation. NeurIPS Datasets and Benchmarks, 2021. [arXiv:2203.06823](https://arxiv.org/abs/2203.06823)
7. Ilse M, Tomczak JM, Welling M. Attention-based Deep Multiple Instance Learning. ICML, 2018. [arXiv:1802.04712](https://arxiv.org/abs/1802.04712)
8. Tran D, et al. A Closer Look at Spatiotemporal Convolutions for Action Recognition. CVPR, 2018. [arXiv:1711.11248](https://arxiv.org/abs/1711.11248)
9. Hunter DJ, et al. Evolution of semi-quantitative whole joint assessment of knee OA: MOAKS (MRI Osteoarthritis Knee Score). Osteoarthritis and Cartilage 19(8):990-1002, 2011. [doi:10.1016/j.joca.2011.05.004](https://doi.org/10.1016/j.joca.2011.05.004)
10. Zbontar J, et al. fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. 2018. [arXiv:1811.08839](https://arxiv.org/abs/1811.08839)
11. Bien N, et al. Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet. PLOS Medicine 15(11):e1002699, 2018. [doi:10.1371/journal.pmed.1002699](https://doi.org/10.1371/journal.pmed.1002699)
12. Štajduhar I, et al. Semi-automated detection of anterior cruciate ligament injury from MRI. Computer Methods and Programs in Biomedicine 140:151-164, 2017. [doi:10.1016/j.cmpb.2016.12.006](https://doi.org/10.1016/j.cmpb.2016.12.006)
13. Mei X, et al. RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning. Radiology: Artificial Intelligence 4(5):e210315, 2022. [doi:10.1148/ryai.210315](https://doi.org/10.1148/ryai.210315)

## Appendix

### A. Label sets on the 58 expert studies

| Label set | Macro AUC |
|---|---:|
| [community regex labeler](https://www.kaggle.com/datasets/pilkwang/rsna-knee-report-labels) | 0.812 |
| regex + local Qwen2.5-7B, rank-averaged (on the 52 expert studies released by then) | 0.853 |
| [a public LLM label table](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) (llm_labels_v4_blend) | 0.893 |
| Claude, finding-by-finding read | 0.897 |
| Claude, whole-report-first read | 0.904 |
| labels_v4, LLM read re-adjudicated on 1,041 image- or consensus-disputed reports | 0.906 |
| **final: 0.5 labels_v4 + 0.5 Claude whole-report read** | **0.909** |

Moving to the public LLM table lifted our public score from 0.862 to 0.903, our largest single jump.

### B. Window-MIL stack

64 slices per study from up to five series roles (18, 14, 12, 8, 12), 42 shared windows, arms rank-averaged with equal weights. CoAtNet arms start from ImageNet weights, not the Raptor checkpoint.

| Arm | Backbone | Input | Training data and targets |
|---|---|---|---|
| H6 | CoAtNet-2 | 384 px | RSNA + a 65k-study external knee corpus (OAI, fastMRI [10], MRNet [11], SKM-TEA, KMAR-50K, Rijeka KneeMRI [12]), 4 epochs, stack-prediction targets |
| H6, seed 2 | CoAtNet-2 | 384 px | as H6, second seed |
| C | CoAtNet-2 | 384 px | the Raptor author's CC0 LLM labels |
| CNX | ConvNeXt-V2-B | 384 px | a consensus of three label sets |
| CNXbig | ConvNeXt-V2-B | 384 px | RSNA + the external corpus, 4 epochs, stack-prediction targets |
| EB1 | EfficientNet-B0 | 288 px | a consensus of three label sets |
| slot41, 2 seeds | CoAtNet-2 | 384 px | stack-prediction targets on a second slice allocation (14, 10, 16, 10, 14), kept as decorrelated members |

### C. SKM-TEA U-Nets and crop readers

[smp](https://github.com/qubvel-org/segmentation_models.pytorch) U-Nets with a ResNet-34 encoder, trained on SKM-TEA's 155 DESS knees [6] with six labels: four cartilage plates and two menisci. The side call comes from anatomy and matches the DICOM laterality tag in 98.8% of tagged RSNA studies.

| U-Net | Training data | Held-out Dice |
|---|---|---:|
| sagittal | DESS echoes mixed, gamma and noise, 384 px at 0.417 mm/px | 0.845 |
| coronal | the same volumes resliced, random left-right flips | 0.806 |
| axial | the same volumes resliced | 0.819 |

Crop readers: ConvNeXt-Tiny (ImageNet-22k) on the 8 slices with the most label area, 224 px, mean + max pooled, 5 folds × 12 epochs on the public label table, loss on visible findings only.

| Member | Plane | Window | Findings | Weight |
|---|---|---|---|---:|
| crop v1 | sagittal | 56 mm on each compartment | both menisci, Medial and Lateral OA | 0.3 |
| crop v2 | sagittal | 64 mm notch; 64 mm patellofemoral | ACL; PF OA | 0.5 |
| crop v3 | sagittal | 80 mm, medial edge, shifted posterior | Baker's | 0.3 |
| coronal | coronal | 56 mm on each compartment | both menisci, Medial and Lateral OA | 0.5 |
| coronal MCL | coronal | the coronal crops, MCL head | MCL | 0.3 |
| two-plane | both | 8 sagittal + 8 coronal crops per compartment, one network | both menisci, Medial and Lateral OA | 0.2 |
| axial | axial | 8 × 80 mm on the patellar cartilage | PF OA, Effusion, Synovitis | 0.3 |

### D. Other specialists, referee and sagittal CNN pair

| Member | Model | Input | Findings | Weight |
|---|---|---|---|---:|
| ZC-P | ConvNeXt-V2-Tiny + attention MIL, 3 seeds × 5 folds | sagittal posterior crop | Synovitis | 0.3 |
| ZC-X | ConvNeXt-V2-Tiny + attention MIL, 5 folds | axial patellofemoral crop, 224 × 320 | Synovitis | 0.3 |
| MV-ARM | ConvNeXt-V2-Tiny per region crop, region and slice embeddings, 2-layer transformer, 3 seeds | crops planned by SKM-TEA segmenters on all three planes | both menisci, Synovitis, Contusion | 0.3 |
| BIG-3D | torchvision r3d_18 shared over three whole stacks, 2-layer transformer, 2 seeds | sagittal, coronal and axial stacks at 32 × 224 × 224 | Lateral Meniscus, PF OA, Baker's, Contusion | 0.3 |
| Hoffa (OAI) | the sagittal ConvNeXt recipe | 16 sagittal slices | Synovitis | 0.3 |
| M/L referee | U-Net side calls, a position-aware sagittal reader, ConvNeXt-V2-Tiny + attention MIL, 3 seeds | compartment crops | Medial and Lateral OA, both menisci | λ 0.1 |
| PF referee | ConvNeXt-V2-Tiny + attention MIL | anterior crop | PF OA | λ 0.2 |
| wCNX | ConvNeXt-Tiny, top-k per finding (3 Baker's, Contusion, Fracture; 6 ligaments, menisci; 12 Effusion, Synovitis; 16 OA), 5 folds | 16 sagittal slices, 384 px, 130 mm window | all | 0.3 of the fuse, with wPW16 |
| wPW16 | ConvNeXt-Tiny, learned medial-to-lateral pooling, 5 folds, retrained on the blended labels | as wCNX | all | 0.3 of the fuse, with wCNX |

### E. What did not help

Each change is against the same model without it. The private rows are late submissions against the final pick's 0.947; a range means several submissions.

| Idea | Measured on | Change in score |
|---|---|---:|
| RadImageNet [13] instead of ImageNet weights | expert set | -0.062 |
| suprapatellar pouch crops instead of the axial crop, Synovitis | expert set | -0.029 |
| distilling a teacher into the expert folds it had already seen (+0.016 on the expert set) | public | -0.006 |
| warping knees into an atlas space instead of cropping each knee | out-of-fold | -0.004 |
| 3D members trained on OAI MOAKS grades | private | -0.003 to 0.000 |
| a 9-model 3D ensemble | private | -0.002 to 0.000 |

The 3D ensemble lost only when spread to 9 to 12 findings or when a member ran out of memory. On its original findings it scored 0.000 private and +0.001 public.
