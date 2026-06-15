"""train_logos_asl_citizen.py — fine-tune the Logos MViTv2-S backbone on ASL-Citizen.

Self-contained PyTorch trainer (NO SignCLIP/fairseq). Only depends on mmaction2 for
the MViT backbone module, and reuses the exact preprocessing of extract_logos_features.py
so that features re-extracted afterwards (with that same script) are valid.

Must be run in the logos conda env (mmaction2 installed), from the mmaction2 dir.

Three primary runs (see jobs/): all fine-tune the backbone + a NEW Linear(768->2731) head.
  Run 2 (baseline)     : CE only, original train videos.
      python train_logos_asl_citizen.py --train_csv splits/train.csv \
          --video_dir .../videos --checkpoint data/model/logos_autsl_wlasl_model.pth \
          --save_dir runs/run2 --unfreeze_blocks 2 --epochs 10
  Run 3 (aug-as-data)  : CE only, original + augmented (splits_aug/train.csv).
      ... --train_csv splits_aug/train.csv --aug_frames_dir .../augmented_frames ...
  Run 1 (PRIMARY)      : CE + lambda * appearance-consistency (pull original<->its augs).
      ... --train_csv splits/train.csv --paired --aug_frames_dir .../augmented_frames \
          --lambda_consist 0.5 ...
  Run 4 (optional)     : add --lambda_supcon 0.1 (gloss supervised-contrastive term).

Fine-tune extent: --unfreeze_blocks N (top N of 16 MViT blocks + head). N>=16 = full FT,
which also enables layer-wise LR decay via --llrd.

The checkpoint is saved as {'state_dict': {'backbone.<k>':..., 'head.<k>':...}} so the
UNMODIFIED extract_logos_features.py re-extracts cleanly (it filters the 'backbone.' prefix).
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Preprocessing constants (COPIED from extract_logos_features.py — KEEP IN SYNC) ──
CLIP_LEN       = 32     # frames fed to MViTv2-S
FRAME_INTERVAL = 2      # sample every 2nd frame → each clip spans 64 consecutive frames
INPUT_SIZE     = 224    # spatial size expected by MViTv2-S
RESIZE         = 300    # only used by the 'resize300crop' preprocessing mode
MEAN = np.array([140.99762122, 129.92701646, 125.25081198], dtype=np.float32)
STD  = np.array([62.07248248,  62.94645644,  61.42221137],  dtype=np.float32)

NUM_BLOCKS  = 16        # MViTv2-S has 16 transformer blocks (verified)
AUG_NAMES   = ("glasses", "shirt_1")   # appearance-augmentation variants on disk
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ──────────────────────────────────────────────────────────────────────────────
# Frame loading + clip sampling
# ──────────────────────────────────────────────────────────────────────────────

def _decode_mp4(path):
    """Decode a video file → list of RGB uint8 frames (H, W, 3)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _load_jpg_dir(path):
    """Load a directory of *.jpg frames (sorted) → list of RGB uint8 frames."""
    import cv2
    frames = []
    for fp in sorted(Path(path).glob("*.jpg")):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


PAD_VALUE = 114   # grey, matches Logos SquarePadding default


def _preprocess_frame(frame, mode):
    """Resize + normalize one RGB uint8 frame → (3, 224, 224) float32.

    MUST stay in sync with extract_logos_features.preprocess_frame so that features
    re-extracted after fine-tuning match what the model saw during training.

    mode='logos_native' : resize LONG side -> 300, pad to 300x300 (grey 114),
                          center-crop 224. Aspect-preserving — what the Logos model was
                          trained on; correct for non-square ASL-Citizen. DEFAULT.
    mode='direct224'    : resize directly to 224x224 (legacy; aspect-distorts non-square
                          frames — used by the old baseline/endanchor features).
    mode='resize300crop': resize SHORTEST side -> 300, pad, center-crop 224 (the old aug
                          extractor's path; crops the long-axis edges).
    """
    import cv2
    if mode == "direct224":
        frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    elif mode in ("logos_native", "resize300crop"):
        h, w = frame.shape[:2]
        # logos_native fits the whole frame (long side -> 300); resize300crop fills (short side).
        scale = RESIZE / (max(h, w) if mode == "logos_native" else min(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_h = max(0, RESIZE - nh)
        pad_w = max(0, RESIZE - nw)
        frame = np.pad(frame, ((pad_h // 2, pad_h - pad_h // 2),
                               (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                       mode="constant", constant_values=PAD_VALUE)
        y0 = (frame.shape[0] - INPUT_SIZE) // 2
        x0 = (frame.shape[1] - INPUT_SIZE) // 2
        frame = frame[y0:y0 + INPUT_SIZE, x0:x0 + INPUT_SIZE]
    else:
        raise ValueError(f"unknown preprocessing mode: {mode}")
    frame = (frame.astype(np.float32) - MEAN) / STD   # (H, W, 3)
    return frame.transpose(2, 0, 1)                    # (3, H, W)


def _sample_clip(frames, train, rng, mode, frame_interval):
    """frames (list of RGB uint8) → ONE clip (3, CLIP_LEN, 224, 224) float32.

    Training samples a random temporal window (jitter); eval/val takes the center
    window. Clip-index formula matches extract_logos_features.py exactly.
    """
    T = len(frames)
    span = CLIP_LEN * frame_interval
    if T <= span:
        start = 0
    else:
        max_start = T - span
        start = rng.randint(0, max_start) if train else max_start // 2
    indices = [min(start + i * frame_interval, T - 1) for i in range(CLIP_LEN)]
    clip = np.stack([_preprocess_frame(frames[i], mode) for i in indices])  # (CLIP_LEN,3,H,W)
    return clip.transpose(1, 0, 2, 3)                                       # (3,CLIP_LEN,H,W)


# ──────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_csv_rows(csv_path):
    """Read an ASL-Citizen split CSV → list of (video_file_basename, gloss)."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            basename = os.path.splitext(row["Video file"])[0]
            rows.append((basename, row["Gloss"].strip()))
    return rows


def build_gloss_to_idx(train_csv):
    """Deterministic gloss→idx from the ORIGINAL train split (sorted gloss set)."""
    glosses = sorted({g for _, g in _read_csv_rows(train_csv)})
    return {g: i for i, g in enumerate(glosses)}


def _split_aug_suffix(basename, aug_names):
    """('1234-MARCH_glasses', ('glasses','shirt_1')) -> ('1234-MARCH','glasses').

    Returns (video_id, None) if no known augmentation suffix is present.
    """
    for a in aug_names:
        if basename.endswith("_" + a):
            return basename[: -(len(a) + 1)], a
    return basename, None


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

class ClipDataset(Dataset):
    """One clip per item, for CE-only training (Runs 2 and 3).

    Resolves each CSV row to a frame source: an original mp4 in video_dir, or — if the
    row carries an augmentation suffix (Run 3 / splits_aug) — the JPG frame directory
    augmented_frames/{video_id}/{aug_name}/.
    """

    def __init__(self, csv_path, video_dir, aug_frames_dir, gloss_to_idx, train,
                 orig_mode, aug_mode, frame_interval, aug_frame_interval,
                 aug_names, max_videos=None):
        self.train = train
        self.orig_mode, self.aug_mode = orig_mode, aug_mode
        self.frame_interval, self.aug_frame_interval = frame_interval, aug_frame_interval
        self.items = []   # (kind, path, label, frame_interval, mode)
        missing = 0
        for basename, gloss in _read_csv_rows(csv_path):
            if gloss not in gloss_to_idx:
                continue
            label = gloss_to_idx[gloss]
            video_id, aug = _split_aug_suffix(basename, aug_names)
            if aug is None:
                path = self._find_video(video_dir, basename)
                if path is None:
                    missing += 1
                    continue
                self.items.append(("mp4", path, label, frame_interval, orig_mode))
            else:
                fdir = Path(aug_frames_dir) / video_id / aug
                if not fdir.is_dir():
                    missing += 1
                    continue
                self.items.append(("frames", str(fdir), label, aug_frame_interval, aug_mode))
            if max_videos and len(self.items) >= max_videos:
                break
        print(f"[ClipDataset] {csv_path}: {len(self.items)} items "
              f"({missing} unresolved), train={train}")

    @staticmethod
    def _find_video(video_dir, basename):
        for ext in (".mp4", ".MP4", ".avi", ".mov", ".mkv", ".webm"):
            p = Path(video_dir) / (basename + ext)
            if p.exists():
                return str(p)
        return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        # Skip empty/corrupt videos (0 decoded frames) by advancing to the next item,
        # rather than crashing the whole run on one bad file.
        for _ in range(50):
            kind, path, label, fi, mode = self.items[i]
            frames = _decode_mp4(path) if kind == "mp4" else _load_jpg_dir(path)
            if frames:
                rng = (random.Random((os.getpid() << 20) ^ (i << 4) ^ (int(time.time() * 1e3) & 0xFFFF))
                       if self.train else random.Random(0))
                clip = _sample_clip(frames, self.train, rng, mode, fi)
                return torch.from_numpy(clip), label
            print(f"[ClipDataset] WARNING: 0 frames, skipping {path}", flush=True)
            i = (i + 1) % len(self.items)
        raise RuntimeError("ClipDataset: 50 consecutive empty videos — check the data paths")


class PairedDataset(Dataset):
    """Original clip + its present augmentation clips, for Run 1 (CE + consistency).

    Iterates the ORIGINAL train rows (covers all videos). For each, attaches whichever
    of augmented_frames/{video_id}/{aug_name}/ exist (0, 1 or 2 variants).
    """

    def __init__(self, csv_path, video_dir, aug_frames_dir, gloss_to_idx,
                 orig_mode, aug_mode, frame_interval, aug_frame_interval,
                 aug_names, max_videos=None, require_augs=False):
        self.video_dir = video_dir
        self.orig_mode, self.aug_mode = orig_mode, aug_mode
        self.frame_interval, self.aug_frame_interval = frame_interval, aug_frame_interval
        self.items = []   # (orig_path, [aug_frame_dirs], label)
        n_with_aug = 0
        missing = 0
        for basename, gloss in _read_csv_rows(csv_path):
            if gloss not in gloss_to_idx:
                continue
            video_id, aug = _split_aug_suffix(basename, aug_names)
            if aug is not None:
                continue   # skip aug rows if a splits_aug csv was passed by mistake
            orig_path = ClipDataset._find_video(video_dir, basename)
            if orig_path is None:
                missing += 1
                continue
            aug_dirs = []
            for a in aug_names:
                fdir = Path(aug_frames_dir) / basename / a
                if fdir.is_dir() and any(fdir.glob("*.jpg")):
                    aug_dirs.append(str(fdir))
            if aug_dirs:
                n_with_aug += 1
            elif require_augs:
                continue   # keep only paired items (used by the smoke test)
            self.items.append((orig_path, aug_dirs, gloss_to_idx[gloss]))
            if max_videos and len(self.items) >= max_videos:
                break
        print(f"[PairedDataset] {csv_path}: {len(self.items)} originals "
              f"({n_with_aug} with augs, {missing} unresolved, require_augs={require_augs})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        # Skip empty/corrupt originals; drop any aug variant that fails to load (don't crash).
        for _ in range(50):
            orig_path, aug_dirs, label = self.items[i]
            frames = _decode_mp4(orig_path)
            if frames:
                rng = random.Random((os.getpid() << 20) ^ (i << 4) ^ (int(time.time() * 1e3) & 0xFFFF))
                orig = _sample_clip(frames, True, rng, self.orig_mode, self.frame_interval)
                augs = []
                for d in aug_dirs:
                    af = _load_jpg_dir(d)
                    if af:
                        augs.append(torch.from_numpy(
                            _sample_clip(af, True, rng, self.aug_mode, self.aug_frame_interval)))
                    else:
                        print(f"[PairedDataset] WARNING: 0 aug frames, skipping {d}", flush=True)
                return torch.from_numpy(orig), augs, label
            print(f"[PairedDataset] WARNING: 0 frames, skipping {orig_path}", flush=True)
            i = (i + 1) % len(self.items)
        raise RuntimeError("PairedDataset: 50 consecutive empty originals — check the data paths")


def paired_collate(batch):
    """batch of (orig, [augs], label) → batched originals + flattened augs + owner map."""
    origs = torch.stack([b[0] for b in batch])                  # (B,3,T,H,W)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    aug_clips, aug_owner, aug_labels = [], [], []
    for owner, b in enumerate(batch):
        for a in b[1]:
            aug_clips.append(a)
            aug_owner.append(owner)
            aug_labels.append(b[2])
    if aug_clips:
        augs = torch.stack(aug_clips)                            # (M,3,T,H,W)
        aug_owner = torch.tensor(aug_owner, dtype=torch.long)
        aug_labels = torch.tensor(aug_labels, dtype=torch.long)
    else:
        augs = torch.empty(0)
        aug_owner = torch.empty(0, dtype=torch.long)
        aug_labels = torch.empty(0, dtype=torch.long)
    return {"orig": origs, "labels": labels,
            "augs": augs, "aug_owner": aug_owner, "aug_labels": aug_labels}


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def load_backbone_trainable(checkpoint_path, device):
    """Build MViTv2-S and load Logos weights (replicates extract_logos_features.py),
    but kept TRAINABLE (no global no_grad / requires_grad freeze here)."""
    from mmaction.registry import MODELS
    import mmaction.models  # noqa: F401 — registers MViT

    backbone = MODELS.build(dict(
        type="MViT", arch="small", drop_path_rate=0.1, dim_mul_in_attention=False,
    ))
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    backbone_state = {k[len("backbone."):]: v for k, v in state.items()
                      if k.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    non_head_missing = [k for k in missing if "head" not in k]
    if non_head_missing:
        print(f"[load_backbone] WARNING {len(non_head_missing)} backbone keys missing, "
              f"e.g. {non_head_missing[:3]}")
    if unexpected:
        print(f"[load_backbone] {len(unexpected)} unexpected keys ignored")
    return backbone.to(device)


class LogosClassifier(nn.Module):
    """MViTv2-S backbone + new linear classification head. forward → (feat_768, logits)."""

    def __init__(self, checkpoint, num_classes, device):
        super().__init__()
        self.backbone = load_backbone_trainable(checkpoint, device)
        self.head = nn.Linear(768, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.head.to(device)

    def _pool(self, feat):
        while isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.ndim == 5:      # (N,C,T,H,W)
            feat = feat.mean(dim=[2, 3, 4])
        elif feat.ndim == 3:    # (N,seq,C)
            feat = feat.mean(dim=1)
        return feat             # (N,768)

    def forward(self, x):
        feat = self._pool(self.backbone(x))
        return feat, self.head(feat)


# ──────────────────────────────────────────────────────────────────────────────
# Unfreezing + optimizer param groups
# ──────────────────────────────────────────────────────────────────────────────

def _block_index(name):
    """Depth index of a backbone param: patch_embed/pos/cls=0, blocks.i=i+1, else NUM_BLOCKS+1."""
    if name.startswith("blocks."):
        try:
            return int(name.split(".")[1]) + 1
        except (IndexError, ValueError):
            return NUM_BLOCKS + 1
    if name.startswith(("patch_embed", "pos_embed", "cls_token")):
        return 0
    return NUM_BLOCKS + 1   # final norm etc.


def set_trainable(model, unfreeze_blocks):
    """Freeze all but the top `unfreeze_blocks` MViT blocks (head always trainable).

    unfreeze_blocks >= NUM_BLOCKS → full backbone fine-tune.
    Frozen blocks are also kept in eval() mode (disables stochastic depth).
    """
    bb = model.backbone
    full = unfreeze_blocks >= NUM_BLOCKS
    first_trainable = NUM_BLOCKS - unfreeze_blocks   # block idx threshold
    for name, p in bb.named_parameters():
        depth = _block_index(name)
        if full:
            p.requires_grad_(True)
        elif name.startswith("blocks.") and (depth - 1) >= first_trainable:
            p.requires_grad_(True)
        elif depth == NUM_BLOCKS + 1:   # final norm — cheap, let it adapt
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[set_trainable] unfreeze_blocks={unfreeze_blocks} (full={full}) → "
          f"{n_train/1e6:.1f}M / {n_total/1e6:.1f}M params trainable")


def set_train_mode(model, unfreeze_blocks):
    """Put model in train() but keep frozen MViT blocks in eval() (no stochastic depth)."""
    model.train()
    if unfreeze_blocks >= NUM_BLOCKS:
        return
    bb = model.backbone
    first_trainable = NUM_BLOCKS - unfreeze_blocks
    for i, blk in enumerate(bb.blocks):
        if i < first_trainable:
            blk.eval()


def build_param_groups(model, base_lr, llrd, weight_decay):
    """Layer-wise-LR-decayed AdamW param groups (only trainable params).

    lr(depth) = base_lr * llrd^(max_depth - depth). Norm/bias/1-D params get no weight decay.
    llrd=1.0 → uniform lr (used for partial fine-tune).
    """
    max_depth = NUM_BLOCKS + 1
    groups = {}
    for name, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        depth = _block_index(name)
        lr = base_lr * (llrd ** (max_depth - depth))
        wd = 0.0 if p.ndim == 1 else weight_decay
        key = (round(lr, 12), wd)
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": wd})["params"].append(p)
    # head at max depth (= base_lr)
    for p in model.head.parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if p.ndim == 1 else weight_decay
        key = (round(base_lr, 12), wd)
        groups.setdefault(key, {"params": [], "lr": base_lr, "weight_decay": wd})["params"].append(p)
    return list(groups.values())


# ──────────────────────────────────────────────────────────────────────────────
# Optional supervised-contrastive loss (Run 4)
# ──────────────────────────────────────────────────────────────────────────────

def supcon_loss(feats, labels, temperature=0.07):
    """Supervised contrastive loss (Khosla et al.) over L2-normalized features."""
    feats = F.normalize(feats, dim=-1)
    sim = feats @ feats.T / temperature                       # (N,N)
    N = feats.size(0)
    self_mask = torch.eye(N, dtype=torch.bool, device=feats.device)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask   # (N,N)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = pos.sum(1)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return feats.new_zeros(())
    mean_log_prob_pos = (pos * log_prob).sum(1)[valid] / pos_counts[valid]
    return -mean_log_prob_pos.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def cosine_lr(step, total_steps, warmup_steps, base_scale=1.0):
    if step < warmup_steps:
        return base_scale * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_scale * 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))


@torch.no_grad()
def validate(model, loader, device, use_amp):
    """Closed-set classification on the (signer-disjoint) val split.
    Returns (val_ce, val_top1, val_top5). Deterministic center clip per video."""
    model.eval()
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    tot = c1 = c5 = 0
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, logits = model(clips)
        loss_sum += ce_sum(logits.float(), labels).item()
        top5 = logits.topk(5, dim=1).indices
        c1 += (top5[:, 0] == labels).sum().item()
        c5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        tot += labels.numel()
    return loss_sum / max(1, tot), c1 / max(1, tot), c5 / max(1, tot)


# ──────────────────────────────────────────────────────────────────────────────
# MLflow logging (self-contained, failure-tolerant — mirrors mlflow_quickstart.py /
# mlflow_eval_logging.py from the MMPT repo, which can't be imported from here)
# ──────────────────────────────────────────────────────────────────────────────

_MLFLOW_TRACKING_URI = "https://mlflow.ai.mytkhgroup.com/"


class MLflowLogger:
    """Logs backbone-FT training to MLflow. Any failure (missing pkg / unreachable
    server) disables logging and never interrupts training."""

    def __init__(self, enable, experiment, run_name, params):
        self.ok = False
        self.mlflow = None
        self.run_id = None
        if not enable:
            return
        try:
            import mlflow
            self.mlflow = mlflow
            mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment)
            run = mlflow.start_run(run_name=run_name)
            self.run_id = run.info.run_id
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.set_tags({"kind": "backbone_ft",
                             "node": os.environ.get("SLURMD_NODENAME", "unknown"),
                             "job_id": os.environ.get("SLURM_JOB_ID", "unknown")})
            self.ok = True
            print(f"[MLflow] logging to '{experiment}' run '{run_name}'")
        except Exception as exc:
            print(f"[MLflow] disabled ({exc})")

    def log(self, metrics, step=None):
        if not self.ok:
            return
        try:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
        except Exception as exc:
            print(f"[MLflow] log warning: {exc}")

    def log_artifact_file(self, path):
        """(Re)upload a file as an artifact — overwrites the previous copy, so calling
        it periodically gives a live-updating log in the MLflow UI."""
        if not self.ok or not path:
            return
        try:
            if os.path.exists(path):
                self.mlflow.log_artifact(path)
        except Exception as exc:
            print(f"[MLflow] artifact warning: {exc}", flush=True)

    def end(self):
        if not self.ok:
            return
        try:
            self.mlflow.end_run()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--video_dir", required=True, help="dir with original ASL-Citizen mp4s")
    ap.add_argument("--aug_frames_dir", default=None,
                    help="augmented_frames/ root ({video_id}/{aug_name}/*.jpg)")
    ap.add_argument("--checkpoint", required=True, help="logos_autsl_wlasl_model.pth")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--num_classes", type=int, default=2731)
    ap.add_argument("--aug_names", nargs="+", default=list(AUG_NAMES))
    # preprocessing modes
    ap.add_argument("--orig_preproc", choices=["logos_native", "direct224", "resize300crop"],
                    default="logos_native",
                    help="must match re-extraction (extract_logos_features.py --preproc)")
    ap.add_argument("--aug_preproc", choices=["logos_native", "direct224", "resize300crop"],
                    default="logos_native")
    ap.add_argument("--frame_interval", type=int, default=FRAME_INTERVAL)
    ap.add_argument("--aug_frame_interval", type=int, default=FRAME_INTERVAL,
                    help="use 1 if augmented frames were generated with stride 2")
    # objective
    ap.add_argument("--paired", action="store_true", help="Run 1: enable consistency loss")
    ap.add_argument("--require_augs", action="store_true",
                    help="(paired) keep only videos that HAVE augmentations — guarantees the "
                         "consistency loss fires; used by the smoke test")
    ap.add_argument("--lambda_consist", type=float, default=0.5)
    ap.add_argument("--lambda_supcon", type=float, default=0.0)
    ap.add_argument("--supcon_temperature", type=float, default=0.07)
    # fine-tune extent
    ap.add_argument("--unfreeze_blocks", type=int, default=2,
                    help=f"top N of {NUM_BLOCKS} MViT blocks to unfreeze; >={NUM_BLOCKS} = full FT")
    ap.add_argument("--llrd", type=float, default=1.0, help="layer-wise LR decay (full FT)")
    # optim
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--clip_grad", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no_amp", action="store_true")
    # validation (signer-disjoint val split → best-checkpoint selection + overfit signal)
    ap.add_argument("--val_csv", default=None,
                    help="val split CSV; if set, log val_ce/val_top1/val_top5 per epoch "
                         "and save checkpoint_best.pt by best val_top1")
    ap.add_argument("--val_every", type=int, default=1, help="validate every N epochs")
    ap.add_argument("--val_batch_size", type=int, default=32)
    ap.add_argument("--val_max_videos", type=int, default=None,
                    help="cap val videos for speed (default: full val split)")
    # misc / smoke
    ap.add_argument("--max_videos", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 20 videos, 2 steps, batch 2 — verifies the pipeline")
    # mlflow
    ap.add_argument("--mlflow", action="store_true",
                    help="log training (CE / consistency / lr) to MLflow; failure-tolerant")
    ap.add_argument("--mlflow_experiment", type=str, default="asl-citizen-backbone-ft")
    ap.add_argument("--mlflow_run_name", type=str, default=None,
                    help="MLflow run name (default: basename of --save_dir, e.g. asl_ft_run1)")
    ap.add_argument("--slurm_log", default=None,
                    help="path to the SLURM .txt log; re-uploaded to MLflow every "
                         "--artifact_every steps for live monitoring")
    ap.add_argument("--artifact_every", type=int, default=500,
                    help="upload the --slurm_log artifact every N optimizer steps")
    args = ap.parse_args()

    if args.smoke:
        args.max_videos = args.max_videos or 20
        args.max_steps = args.max_steps or 2
        args.batch_size = min(args.batch_size, 2)
        args.epochs = 1
        args.workers = min(args.workers, 2)

    if (args.paired or "splits_aug" in args.train_csv) and not args.aug_frames_dir:
        ap.error("--aug_frames_dir is required for --paired or splits_aug training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    print(f"Device: {device} | AMP: {use_amp} | paired: {args.paired} | "
          f"lambda_consist: {args.lambda_consist} | lambda_supcon: {args.lambda_supcon}")

    # gloss vocab (always from the ORIGINAL train split for determinism)
    base_train_csv = args.train_csv.replace("splits_aug", "splits")
    gloss_to_idx = build_gloss_to_idx(base_train_csv)
    print(f"gloss vocab: {len(gloss_to_idx)} glosses (from {base_train_csv})")
    with open(os.path.join(args.save_dir, "gloss_to_idx.json"), "w") as f:
        json.dump(gloss_to_idx, f)

    # dataset / loader
    if args.paired:
        ds = PairedDataset(args.train_csv, args.video_dir, args.aug_frames_dir, gloss_to_idx,
                           args.orig_preproc, args.aug_preproc,
                           args.frame_interval, args.aug_frame_interval,
                           args.aug_names, args.max_videos, require_augs=args.require_augs)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, collate_fn=paired_collate,
                            drop_last=True, pin_memory=True)
    else:
        ds = ClipDataset(args.train_csv, args.video_dir, args.aug_frames_dir, gloss_to_idx,
                         True, args.orig_preproc, args.aug_preproc,
                         args.frame_interval, args.aug_frame_interval,
                         args.aug_names, args.max_videos)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=True, pin_memory=True)
    if len(ds) == 0:
        raise RuntimeError("empty dataset — check paths/splits")

    # validation loader (originals only, deterministic center clip)
    val_loader = None
    if args.val_csv:
        val_ds = ClipDataset(args.val_csv, args.video_dir, args.aug_frames_dir, gloss_to_idx,
                             False, args.orig_preproc, args.aug_preproc,
                             args.frame_interval, args.aug_frame_interval,
                             args.aug_names, args.val_max_videos)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

    # model
    model = LogosClassifier(args.checkpoint, args.num_classes, device)
    set_trainable(model, args.unfreeze_blocks)

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs if not args.max_steps else args.max_steps
    warmup_steps = int(total_steps * args.warmup_frac)

    param_groups = build_param_groups(model, args.lr, args.llrd, args.weight_decay)
    base_lrs = [g["lr"] for g in param_groups]
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.98))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce = nn.CrossEntropyLoss()

    print(f"steps/epoch={steps_per_epoch} total_steps={total_steps} warmup={warmup_steps}")

    mlf = MLflowLogger(
        args.mlflow, args.mlflow_experiment,
        args.mlflow_run_name or os.path.basename(os.path.normpath(args.save_dir)),
        params={"paired": args.paired, "lambda_consist": args.lambda_consist,
                "lambda_supcon": args.lambda_supcon, "unfreeze_blocks": args.unfreeze_blocks,
                "llrd": args.llrd, "lr": args.lr, "weight_decay": args.weight_decay,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "grad_accum": args.grad_accum, "aug_names": ",".join(args.aug_names),
                "orig_preproc": args.orig_preproc, "aug_preproc": args.aug_preproc,
                "train_csv": args.train_csv, "num_classes": args.num_classes,
                "total_steps": total_steps, "n_train": len(ds)},
    )
    # Expose the run_id so the SLURM job can attach the .txt log as an artifact at the end.
    if mlf.run_id:
        try:
            with open(os.path.join(args.save_dir, "mlflow_run_id.txt"), "w") as f:
                f.write(mlf.run_id)
        except Exception:
            pass

    gstep = 0
    n_pair_batches = 0
    best_val = -1.0
    best_path = os.path.join(args.save_dir, "checkpoint_best.pt")
    last_path = os.path.join(args.save_dir, "checkpoint_last.pt")

    def save_ckpt(path, extra=None):
        state = {}
        for k, v in model.backbone.state_dict().items():
            state[f"backbone.{k}"] = v
        for k, v in model.head.state_dict().items():
            state[f"head.{k}"] = v
        torch.save({"state_dict": state, "gloss_to_idx": gloss_to_idx,
                    "args": vars(args), **(extra or {})}, path)

    stop = False
    for epoch in range(args.epochs):
        set_train_mode(model, args.unfreeze_blocks)
        opt.zero_grad(set_to_none=True)
        running, seen = 0.0, 0
        t0 = time.time()
        for it, batch in enumerate(loader):
            lr_scale = cosine_lr(gstep, total_steps, warmup_steps)
            for g, blr in zip(opt.param_groups, base_lrs):
                g["lr"] = blr * lr_scale

            with torch.cuda.amp.autocast(enabled=use_amp):
                if args.paired:
                    orig = batch["orig"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    B = orig.size(0)
                    M = batch["augs"].size(0) if batch["augs"].ndim == 5 else 0
                    if M > 0:
                        x = torch.cat([orig, batch["augs"].to(device, non_blocking=True)], 0)
                    else:
                        x = orig
                    feat, logits = model(x)
                    if M > 0:
                        labels_all = torch.cat(
                            [labels, batch["aug_labels"].to(device, non_blocking=True)], 0)
                    else:
                        labels_all = labels
                    loss_ce = ce(logits, labels_all)
                    loss = loss_ce
                    if M > 0:
                        owner = batch["aug_owner"].to(device, non_blocking=True)
                        cos = F.cosine_similarity(feat[B:], feat[:B][owner], dim=-1)
                        loss_consist = (1.0 - cos).mean()
                        loss = loss + args.lambda_consist * loss_consist
                        n_pair_batches += 1
                    else:
                        loss_consist = torch.zeros((), device=device)
                    if args.lambda_supcon > 0:
                        loss = loss + args.lambda_supcon * supcon_loss(
                            feat, labels_all, args.supcon_temperature)
                else:
                    clips, labels = batch
                    clips = clips.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    feat, logits = model(clips)
                    loss_ce = ce(logits, labels)
                    loss = loss_ce
                    loss_consist = torch.zeros((), device=device)
                    if args.lambda_supcon > 0:
                        loss = loss + args.lambda_supcon * supcon_loss(
                            feat, labels, args.supcon_temperature)

                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (it + 1) % args.grad_accum == 0:
                if args.clip_grad:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                gstep += 1

                running += loss_ce.item(); seen += 1
                if gstep % args.log_interval == 0:
                    ce_avg = running / seen
                    cur_lr = base_lrs[0] * lr_scale
                    print(f"ep{epoch} step{gstep}/{total_steps} "
                          f"ce={ce_avg:.4f} consist={loss_consist.item():.4f} "
                          f"lr={cur_lr:.2e} "
                          f"{seen*args.batch_size/(time.time()-t0):.1f} clip/s", flush=True)
                    logd = {"train_ce": ce_avg, "lr": cur_lr, "epoch": epoch}
                    if args.paired:                       # consist is only meaningful when paired
                        logd["consist"] = loss_consist.item()
                    mlf.log(logd, step=gstep)
                    running, seen, t0 = 0.0, 0, time.time()

                # refresh the live SLURM-log artifact in MLflow (coarser than metric logging)
                if args.slurm_log and gstep % args.artifact_every == 0:
                    mlf.log_artifact_file(args.slurm_log)

                if args.max_steps and gstep >= args.max_steps:
                    stop = True
                    break
        save_ckpt(last_path, extra={"epoch": epoch, "gstep": gstep})

        if val_loader is not None and ((epoch + 1) % args.val_every == 0
                                       or epoch == args.epochs - 1):
            vce, vt1, vt5 = validate(model, val_loader, device, use_amp)
            print(f"  [val] epoch {epoch}: ce={vce:.4f} top1={vt1:.4f} top5={vt5:.4f}",
                  flush=True)
            mlf.log({"val_ce": vce, "val_top1": vt1, "val_top5": vt5}, step=gstep)
            if vt1 > best_val:
                best_val = vt1
                save_ckpt(best_path, extra={"epoch": epoch, "val_top1": vt1})
                print(f"  [val] new best top1={vt1:.4f} → checkpoint_best.pt", flush=True)

        if stop:
            break

    save_ckpt(last_path, extra={"epoch": args.epochs, "gstep": gstep})
    if best_val < 0:        # no validation ran → best = last
        save_ckpt(best_path, extra={"note": "alias of last (no validation)"})
    else:
        print(f"Best val top1 = {best_val:.4f} (checkpoint_best.pt)")
    if args.paired:
        print(f"batches with ≥1 aug pair: {n_pair_batches}")
        mlf.log({"n_pair_batches": n_pair_batches}, step=gstep)
    mlf.log_artifact_file(args.slurm_log)   # final training-log snapshot
    mlf.end()
    print(f"Done. Checkpoints in {args.save_dir}")
    print(f"Re-extract with: python extract_logos_features.py --checkpoint {last_path} ...")


if __name__ == "__main__":
    main()
