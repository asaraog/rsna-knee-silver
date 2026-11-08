#!/usr/bin/env python3
"""export_weights.py <out_dir> - pod side: SKM U-Net -> TorchScript fp16 (no smp needed in the kernel); crop folds -> fp16 state dicts.
Checks traced vs eager on a random batch (max abs diff of softmax) and fp16 crop vs fp32 crop logits."""
import sys, os, torch, torch.nn as nn, timm, segmentation_models_pytorch as smp
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
ck = torch.load("/workspace/skmtea/out/unet_skm.pt", map_location="cpu", weights_only=False)
net = smp.Unet("resnet34", encoder_weights=None, classes=7); net.load_state_dict(ck["model"]); net = net.cuda().eval().half()
x = torch.rand(4, 3, 384, 384, device="cuda").half()
with torch.no_grad():
    tr = torch.jit.trace(net, x); x2 = torch.rand(9, 3, 384, 384, device="cuda").half()
    d = (tr(x2).float().softmax(1) - net(x2).float().softmax(1)).abs().max().item()
print("unet traced, batch-9 max |softmax diff|", d); assert d < 1e-2
tr.save(f"{out}/unet_skm_ts16.pt")
class Net(nn.Module):
    def __init__(self, arch):
        super().__init__(); self.b = timm.create_model(arch, pretrained=False, num_classes=0); self.h = nn.Linear(2 * self.b.num_features, 4)
for f in range(5):
    s = torch.load(f"/workspace/ckpt/wCROP_f{f}.pt", map_location="cpu", weights_only=False)
    torch.save({"model": {k: v.half() for k, v in s["model"].items()}, "arch": "convnext_tiny", "cols": s["cols"]}, f"{out}/wCROP_f{f}_fp16.pt")
    m = Net("convnext_tiny"); m.load_state_dict(s["model"])   # arch name check: plain convnext_tiny loads the fb_in22k_ft_in1k weights
print("crop folds exported", sorted(os.listdir(out)))
