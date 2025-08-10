# deepdta_infer_all.py
import os, io, math, json, hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =========================
# Model (given)
# =========================
class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, channels, kernels, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        convs = []
        in_ch = emb_dim
        for k, ch in zip(kernels, channels):
            convs.append(nn.Sequential(
                nn.Conv1d(in_ch, ch, kernel_size=k, padding=k//2),
                nn.ReLU(),
                nn.BatchNorm1d(ch)
            ))
            in_ch = ch
        self.convs = nn.ModuleList(convs)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):           # (B, L)
        x = self.emb(x)             # (B, L, E)
        x = x.transpose(1, 2)       # (B, E, L)
        for block in self.convs:
            x = block(x)
        x = torch.max(x, dim=2).values   # (B, C)
        return self.drop(x)

class DeepDTARegressor(nn.Module):
    def __init__(self,
                 smi_vocab_size, prot_vocab_size,
                 emb_smi=64, emb_prot=64,
                 smi_channels=(128,128,128), prot_channels=(256,256,256),
                 smi_kernels=(5,7,11), prot_kernels=(7,11,15),
                 hidden=512, dropout=0.2):
        super().__init__()
        self.smi_enc  = CNNEncoder(smi_vocab_size,  emb_smi,  smi_channels,  smi_kernels,  dropout)
        self.prot_enc = CNNEncoder(prot_vocab_size, emb_prot, prot_channels, prot_kernels, dropout)
        fused = smi_channels[-1] + prot_channels[-1]
        self.mlp = nn.Sequential(
            nn.Linear(fused, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, xs, xp):
        hs = self.smi_enc(xs)
        hp = self.prot_enc(xp)
        return self.mlp(torch.cat([hs, hp], dim=1))

# =========================
# Load checkpoint
# =========================
CKPT_PATH = os.environ.get("CKPT_PATH", "best_model#30.pt")  # adjust if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)


SMI_VOCAB: Dict[str, int] = ckpt["smi_vocab"]
PROT_VOCAB: Dict[str, int] = ckpt["prot_vocab"]
y_mean: float = float(ckpt["y_mean"])
y_std: float  = float(ckpt["y_std"])
CONF = ckpt.get("config", {})
MAX_SMI = int(CONF.get("max_len_smi", 200))
MAX_PROT = int(CONF.get("max_len_prot", 1000))

model = DeepDTARegressor(
    smi_vocab_size=len(SMI_VOCAB),
    prot_vocab_size=len(PROT_VOCAB),
    emb_smi=64, emb_prot=64,
    smi_channels=(128,128,128),
    prot_channels=(256,256,256),
    smi_kernels=(5,7,11),
    prot_kernels=(7,11,15),
    hidden=512, dropout=0.2
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

PAD_SMI = SMI_VOCAB["<pad>"]
PAD_PROT = PROT_VOCAB["<pad>"]
UNK_SMI = SMI_VOCAB["<unk>"]
UNK_PROT = PROT_VOCAB["<unk>"]

REQ_COLS = {"Ligand SMILES", "BindingDB Target Chain Sequence 1"}

# =========================
# Helpers
# =========================
UNC_HIGH = 0.5
PAD_WARN_SMI = 0.7
PAD_WARN_PROT = 0.95
OOV_WARN = 0.01

def encode(text: str, vocab: Dict[str,int], max_len: int, pad_id: int, unk_id: int):
    s = str(text)
    clipped = len(s) > max_len
    s_used = s[:max_len]
    ids = [vocab.get(ch, unk_id) for ch in s_used]
    oov = sum(1 for ch in s_used if ch not in vocab)
    if len(ids) < max_len:
        ids += [pad_id]*(max_len - len(ids))
    pad_frac = float(ids.count(pad_id)) / max_len if max_len > 0 else 0.0
    oov_rate = float(oov) / max(1, len(s_used))
    return np.array(ids, dtype=np.int64), {
        "clipped": clipped, "pad_frac": pad_frac, "oov_rate": oov_rate,
        "orig_len": len(s), "used_len": len(s_used),
    }

def kd_from_p(p: float):
    kd_M = 10.0**(-p)
    return {"Kd_M": kd_M, "Kd_uM": kd_M*1e6, "Kd_nM": kd_M*1e9}

def strength_bucket(p: float):
    if p >= 9:  return "very strong", 95
    if p >= 8:  return "strong", 80
    if p >= 7:  return "moderate", 60
    if p >= 6:  return "weak", 40
    return "very weak", 20

def percentile_from_gaussian(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0: return 50.0
    z = (x - mu) / sigma
    return 0.5*(1.0 + math.erf(z / math.sqrt(2.0))) * 100.0

def mc_dropout_predict(xs_t: torch.Tensor, xp_t: torch.Tensor, T: int = 20):
    model.train()
    outs = []
    with torch.no_grad():
        for _ in range(T):
            outs.append(model(xs_t, xp_t).squeeze(1).cpu().numpy())
    model.eval()
    outs = np.stack(outs, axis=0)
    mean_z = outs.mean(axis=0); std_z = outs.std(axis=0)
    mean = mean_z * y_std + y_mean
    std  = std_z  * y_std
    return mean, std

def confidence_flag(std: float, diag: dict) -> str:
    flags = []
    if std > UNC_HIGH: flags.append("high_uncertainty")
    if diag["smi_pad_frac"] > PAD_WARN_SMI: flags.append("smi_heavily_padded")
    if diag["prot_pad_frac"] > PAD_WARN_PROT: flags.append("prot_heavily_padded")
    if diag["smi_oov_rate"] > OOV_WARN: flags.append("smi_oov")
    if diag["prot_oov_rate"] > OOV_WARN: flags.append("prot_oov")
    if diag["smi_clipped"]: flags.append("smi_clipped")
    if diag["prot_clipped"]: flags.append("prot_clipped")
    return ",".join(flags) if flags else "ok"

# =========================
# Core inference
# =========================
def predict_from_df(df: pd.DataFrame, mc_passes: int = 20, return_summary: bool = True) -> Dict[str, Any]:
    if not REQ_COLS.issubset(df.columns):
        missing = list(REQ_COLS - set(df.columns))
        raise ValueError(f"missing columns: {missing}")

    smi = df["Ligand SMILES"].astype(str).tolist()
    prot = df["BindingDB Target Chain Sequence 1"].astype(str).tolist()

    xs_list, xp_list, diag_list = [], [], []
    dup = {}
    for i, (s, p) in enumerate(zip(smi, prot)):
        ids_smi, d_smi = encode(s, SMI_VOCAB, MAX_SMI, PAD_SMI, UNK_SMI)
        ids_prot, d_prot = encode(p, PROT_VOCAB, MAX_PROT, PAD_PROT, UNK_PROT)
        xs_list.append(ids_smi); xp_list.append(ids_prot)
        h = hashlib.sha256((s+"|"+p).encode()).hexdigest()
        dup.setdefault(h, []).append(i)
        diag_list.append({
            "smi_clipped": d_smi["clipped"], "prot_clipped": d_prot["clipped"],
            "smi_pad_frac": d_smi["pad_frac"], "prot_pad_frac": d_prot["pad_frac"],
            "smi_oov_rate": d_smi["oov_rate"], "prot_oov_rate": d_prot["oov_rate"],
            "smi_len": d_smi["orig_len"], "prot_len": d_prot["orig_len"],
            "smi_used": d_smi["used_len"], "prot_used": d_prot["used_len"],
        })

    xs_t = torch.tensor(np.stack(xs_list), dtype=torch.long, device=device)
    xp_t = torch.tensor(np.stack(xp_list), dtype=torch.long, device=device)

    mean_aff, std_aff = mc_dropout_predict(xs_t, xp_t, T=int(mc_passes))

    results = []
    for i, (m, s, d) in enumerate(zip(mean_aff, std_aff, diag_list)):
        bucket, score = strength_bucket(float(m))
        kd = kd_from_p(float(m))
        pct = percentile_from_gaussian(float(m), y_mean, y_std)
        lo = float(m - 1.96 * s)
        hi = float(m + 1.96 * s)
        conf = confidence_flag(float(s), d)
        results.append({
            "index": i,
            "pAff_pred": float(m),
            "pAff_uncertainty": float(s),
            "ci95_low": lo,
            "ci95_high": hi,
            "confidence": conf,
            "percentile_vs_train": float(pct),
            "strength_bucket": bucket,
            "strength_score": int(score),
            "Kd_M": kd["Kd_M"], "Kd_uM": kd["Kd_uM"], "Kd_nM": kd["Kd_nM"],
            **d
        })

    duplicates = [v for v in dup.values() if len(v) > 1]

    summary = None
    if return_summary and results:
        vals = np.array([r["pAff_pred"] for r in results])
        uncs = np.array([r["pAff_uncertainty"] for r in results])
        cats = {"very strong":0,"strong":0,"moderate":0,"weak":0,"very weak":0}
        for r in results: cats[r["strength_bucket"]] += 1
        summary = {
            "n": len(results),
            "pAff_mean": float(vals.mean()),
            "pAff_std": float(vals.std()),
            "uncertainty_mean": float(uncs.mean()),
            "strength_counts": cats,
            "fraction_clipped_any": float(np.mean([r["smi_clipped"] or r["prot_clipped"] for r in results])),
            "duplicates": duplicates[:100],
            "checkpoint": os.path.basename(CKPT_PATH),
            "device": str(device),
            "max_len_smi": MAX_SMI,
            "max_len_prot": MAX_PROT,
        }

    return {"results": results, "summary": summary}

def predict_from_file(path: str | Path, sep: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    path_str = str(path)
    if sep is None:
        sep = "\t" if path_str.lower().endswith(".tsv") else ","
    df = pd.read_csv(path_str, sep=sep, low_memory=False)
    return predict_from_df(df, **kwargs)

# =========================
# Plotting + exports
# =========================
def _make_plots(out_df: pd.DataFrame, summary: Dict[str, Any], outdir: str) -> Dict[str, str]:
    paths = {}
    os.makedirs(outdir, exist_ok=True)

    # 1) Histogram of pAff
    plt.figure()
    plt.hist(out_df["pAff_pred"].values, bins=20)
    plt.xlabel("pAff"); plt.ylabel("count"); plt.title("Predicted pAff")
    plt.tight_layout()
    paths["hist_pAff"] = os.path.join(outdir, "hist_pAff.png")
    plt.savefig(paths["hist_pAff"], dpi=150)

    # 2) Scatter: uncertainty vs pAff
    plt.figure()
    plt.scatter(out_df["pAff_pred"].values, out_df["pAff_uncertainty"].values)
    plt.xlabel("pAff"); plt.ylabel("uncertainty (std)"); plt.title("Uncertainty vs pAff")
    plt.tight_layout()
    paths["scatter_uncertainty"] = os.path.join(outdir, "scatter_uncertainty.png")
    plt.savefig(paths["scatter_uncertainty"], dpi=150)

    # 3) Bar: strength buckets
    plt.figure()
    labels = ["very weak","weak","moderate","strong","very strong"]
    counts = [summary["strength_counts"].get(k,0) for k in labels]
    plt.bar(labels, counts)
    plt.xlabel("bucket"); plt.ylabel("count"); plt.title("Strength categories")
    plt.tight_layout()
    paths["bar_strength"] = os.path.join(outdir, "bar_strength.png")
    plt.savefig(paths["bar_strength"], dpi=150)

    # 4) Histogram of percentiles
    plt.figure()
    plt.hist(out_df["percentile_vs_train"].values, bins=20)
    plt.xlabel("percentile vs train"); plt.ylabel("count"); plt.title("Percentiles")
    plt.tight_layout()
    paths["hist_percentiles"] = os.path.join(outdir, "hist_percentiles.png")
    plt.savefig(paths["hist_percentiles"], dpi=150)

    # 5) Padding vs uncertainty
    plt.figure()
    pad_any = np.maximum(out_df["smi_pad_frac"].values, out_df["prot_pad_frac"].values)
    plt.scatter(pad_any, out_df["pAff_uncertainty"].values)
    plt.xlabel("max pad fraction"); plt.ylabel("uncertainty (std)")
    plt.title("Padding vs uncertainty")
    plt.tight_layout()
    paths["scatter_pad_uncertainty"] = os.path.join(outdir, "scatter_pad_uncertainty.png")
    plt.savefig(paths["scatter_pad_uncertainty"], dpi=150)

    return paths

def save_outputs(out: Dict[str, Any], outdir: str, source_path: Optional[str] = None) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    out_df = pd.DataFrame(out["results"])
    pred_csv = os.path.join(outdir, "predictions_enriched.csv")
    out_df.to_csv(pred_csv, index=False)

    summary_json = os.path.join(outdir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(out["summary"], f, indent=2)

    if source_path:
        with open(os.path.join(outdir, "source.txt"), "w") as f:
            f.write(str(source_path))

    fig_paths = _make_plots(out_df, out["summary"], outdir)
    return {"csv": pred_csv, "summary": summary_json, **fig_paths}

# =========================
# One-call wrapper
# =========================
def run_affinity_inference(file_path: str | Path,
                           mc_passes: int = 20,
                           outdir: str = "affinity_run") -> Dict[str, Any]:
    out = predict_from_file(file_path, mc_passes=mc_passes, return_summary=True)
    saved_files = save_outputs(out, outdir=outdir, source_path=str(file_path))
    return {"results": out["results"], "summary": out["summary"], "saved_files": saved_files}

# =========================
# Example CLI usage
# =========================
if __name__ == "__main__":
    test_path = "bindingdb_cleaned_test.csv"   # set your file
    res = run_affinity_inference(test_path, mc_passes=10, outdir="affinity_run")
    print("Summary:", res["summary"])
    print("Saved:", res["saved_files"])
