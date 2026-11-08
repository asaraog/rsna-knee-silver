#!/usr/bin/env python3
"""skm_infer.py - run the SKM-TEA compartment U-Net on RSNA sagittal series straight from the competition zip.

Usage: python3 skm_infer.py <unet_skm.pt> <study_ids.txt|ALL> <out_prefix>
Per study: the sagittal fluid-sensitive series (fat-sat preferred, most slices; falls back to any sagittal).
Each slice is put in the TRAINING orientation (rows superior->inferior, cols anterior->posterior), resampled to
0.4167 mm and centre-cropped/padded to 384 px, volume p1-p99.5 normalised.
Writes <out_prefix>_slices.csv: one row per slice with LPS x, pixel areas (mm^2) of the 6 classes, and the mean normalised
intensity inside each meniscus class (intrasubstance signal = tear evidence on fluid-sensitive images).
Aggregation into per-knee compartment evidence happens offline (skm_evidence.py).
"""
import io, os, sys, zipfile
import numpy as np, pandas as pd, pydicom, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp

ZIP = os.environ.get("RSNA_ZIP", "/workspace/rsna-knee-abnormality-detection.zip")
mp, ids_f, outp = sys.argv[1:4]
ck = torch.load(mp, map_location="cpu", weights_only=False); R = ck.get("R", 384); PX = 160.0 / R
net = smp.Unet("resnet34", encoder_weights=None, classes=7); net.load_state_dict(ck["model"]); net = net.cuda().eval().half()
DIR = os.environ.get("RSNA_DIR")   # Kaggle layout on disk instead of the zip (OAI staged by extract_for_ap.py)
if DIR:
    class _Z:
        def read(self, n): return open(os.path.join(DIR, n), "rb").read()
        def namelist(self): return [os.path.relpath(os.path.join(r, f), DIR) for r, _, fs in os.walk(os.path.join(DIR, "train_series")) for f in fs]
    z = _Z()
else:
    z = zipfile.ZipFile(ZIP)
ser = pd.read_csv(io.BytesIO(z.read("train_series.csv")))
by = {}
for n in z.namelist():
    p = n.split("/")
    if n.endswith(".dcm") and p[0] == "train_series" and len(p) >= 4: by.setdefault((p[1], p[2]), []).append(n)
ids = sorted({k[0] for k in by}) if ids_f == "ALL" else [l.strip() for l in open(ids_f) if l.strip()]
rows, t_done = [], 0
for sid in ids:
    s = ser[(ser.StudyInstanceUID == sid) & (ser.Anatomical_Plane == "Sagittal")].copy()
    s["n"] = [len(by.get((sid, x), [])) for x in s.SeriesInstanceUID]; s = s[s.n > 4]
    if not len(s): continue
    s = s.sort_values(["Fluid_Sensitive", "Fat_Suppression", "n"], ascending=False); se = s.iloc[0].SeriesInstanceUID
    try:
        ds = [pydicom.dcmread(io.BytesIO(z.read(n)), force=True) for n in by[(sid, se)]]
        ds = [d for d in ds if hasattr(d, "ImagePositionPatient") and hasattr(d, "pixel_array")]
        sh = pd.Series([d.pixel_array.shape for d in ds]).mode()[0]; ds = [d for d in ds if d.pixel_array.shape == sh]
        ds.sort(key=lambda d: float(d.ImagePositionPatient[0]))
        iop = np.array(ds[0].ImageOrientationPatient, float); rd, cd = iop[:3], iop[3:]
        vol = np.stack([d.pixel_array.astype(np.float32) for d in ds]); ps = [float(v) for v in ds[0].PixelSpacing]  # [row sp, col sp]
        if abs(rd[1]) >= abs(cd[1]):          # cols run A-P (training layout); rows run S-I
            if rd[1] < 0: vol = vol[:, :, ::-1]   # cols -> posterior (LPS +y)
            if cd[2] > 0: vol = vol[:, ::-1, :]   # rows -> inferior (LPS -z)
        else:                                  # rows run A-P: transpose to [slice, S-I, A-P]
            vol = vol.transpose(0, 2, 1); ps = ps[::-1]
            if cd[1] < 0: vol = vol[:, :, ::-1]
            if rd[2] > 0: vol = vol[:, ::-1, :]
        vol = np.ascontiguousarray(vol)
        lo, hi = np.percentile(vol, [1, 99.5]); vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        t = torch.from_numpy(vol)[:, None].cuda()
        h, w = int(round(vol.shape[1] * ps[0] / PX)), int(round(vol.shape[2] * ps[1] / PX))
        t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        ph, pw = max(R - h, 0), max(R - w, 0); t = F.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        H, W = t.shape[-2:]; t = t[..., (H - R) // 2:(H - R) // 2 + R, (W - R) // 2:(W - R) // 2 + R]
        with torch.no_grad():
            pr = torch.cat([net(t[i:i + 16].repeat(1, 3, 1, 1).half()).float().softmax(1) for i in range(0, len(t), 16)])
        lab = pr.argmax(1); img = t[:, 0]
        a = PX * PX
        for k in range(len(ds)):
            r = {"StudyInstanceUID": sid, "series": se, "k": k, "n": len(ds), "x": float(ds[k].ImagePositionPatient[0]),
                 "lat_hdr": str(getattr(ds[k], "Laterality", "") or getattr(ds[k], "ImageLaterality", "") or ""),
                 "slice_mm": float(getattr(ds[k], "SpacingBetweenSlices", 0) or getattr(ds[k], "SliceThickness", 0) or 0)}
            for c, nm in enumerate(["pat", "fem", "tibM", "tibL", "menM", "menL"], 1):
                m = lab[k] == c; r[f"a_{nm}"] = float(m.sum()) * a; r[f"p_{nm}"] = float(pr[k, c].sum()) * a
                if nm.startswith("men") or nm.startswith("tib"):
                    r[f"i_{nm}"] = float(img[k][m].mean()) if m.any() else np.nan
                    r[f"h_{nm}"] = float((img[k][m] > 0.5).float().mean()) if m.any() else np.nan
            rows.append(r)
    except Exception as e:
        print("FAIL", sid, type(e).__name__, str(e)[:100], flush=True); continue
    t_done += 1
    if t_done % 100 == 0:
        print(f"{t_done}/{len(ids)}", flush=True); pd.DataFrame(rows).to_csv(outp + "_slices.csv", index=False)
pd.DataFrame(rows).to_csv(outp + "_slices.csv", index=False); print("INFER DONE", t_done)
