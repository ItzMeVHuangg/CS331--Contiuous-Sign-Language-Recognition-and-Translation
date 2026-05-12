
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


BASE_PATH    = Path("/datastore/cndt_phungdtm/KLTN_HoangBinh/Dataset")
PHOENIX_ROOT = BASE_PATH / "PHOENIX-2014-T"
FRAMES_ROOT  = PHOENIX_ROOT / "features" / "fullFrame-210x260px"
ANNOT_ROOT   = PHOENIX_ROOT / "annotations" / "manual"


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Bidirectional token <-> index mapping.
    Special tokens are always added first so their indices are fixed:
      0 = <blank>   (CTC blank)
      1 = <pad>
      2 = <unk>
      3 = <bos>
      4 = <eos>
    """
    BLANK = "<blank>"
    PAD   = "<pad>"
    UNK   = "<unk>"
    BOS   = "<bos>"
    EOS   = "<eos>"

    def __init__(self):
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        for tok in [self.BLANK, self.PAD, self.UNK, self.BOS, self.EOS]:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx]   = token
        return self.token2idx[token]

    def build_from_sequences(self, sequences: List[List[str]]):
        for seq in sequences:
            for tok in seq:
                self._add(tok)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.token2idx[self.UNK]
        return [self.token2idx.get(t, unk) for t in tokens]

    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        special = {self.BLANK, self.PAD, self.UNK, self.BOS, self.EOS}
        tokens  = [self.idx2token.get(i, self.UNK) for i in indices]
        if skip_special:
            tokens = [t for t in tokens if t not in special]
        return tokens

    def __len__(self) -> int:
        return len(self.token2idx)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"


# ──────────────────────────────────────────────────────────────────────────────
# Frame transforms
# ──────────────────────────────────────────────────────────────────────────────

def build_transforms(split: str, img_h: int = 224, img_w: int = 224) -> T.Compose:
    """
    Returns an augmentation pipeline appropriate for each split.
    ImageNet mean/std normalization is applied to all splits.
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if split == "train":
        return T.Compose([
            T.Resize((img_h + 20, img_w + 20)),
            T.RandomCrop((img_h, img_w)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.2),
        ])
    else:                                   # dev / test
        return T.Compose([
            T.Resize((img_h, img_w)),
            T.ToTensor(),
            normalize,
        ])


# ──────────────────────────────────────────────────────────────────────────────
# PHOENIX-2014-T Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PhoenixDataset(Dataset):
    """
    PHOENIX-2014-T dataset loader.

    Paths are resolved automatically from the module-level constants
    (FRAMES_ROOT, ANNOT_ROOT) -- no need to pass paths as arguments.

    Args:
        split             : "train" | "dev" | "test"
        gloss_vocab       : Vocabulary for gloss tokens  (used by CSLR)
        text_vocab        : Vocabulary for German tokens (used by SLT)
        max_frames        : fixed clip length; shorter clips are zero-padded,
                            longer clips are truncated from the end
        temporal_stride   : keep every N-th frame  (1 = keep all frames)
        transform         : torchvision transform applied to each frame PIL image
        return_translation: if True, also return tokenised German translation
    """

    def __init__(
        self,
        split: str,
        gloss_vocab: Vocabulary,
        text_vocab: Vocabulary,
        max_frames: int = 256,
        temporal_stride: int = 1,
        transform: Optional[T.Compose] = None,
        return_translation: bool = True,
    ):
        assert split in ("train", "dev", "test"), \
            f"split must be 'train', 'dev', or 'test' -- got '{split}'"

        self.split              = split
        self.gloss_vocab        = gloss_vocab
        self.text_vocab         = text_vocab
        self.max_frames         = max_frames
        self.temporal_stride    = temporal_stride
        self.transform          = transform
        self.return_translation = return_translation

        # Validate that paths exist before doing anything else
        self._check_paths()

        self.samples = self._load_annotations()
        print(
            f"[PhoenixDataset] split={split:5s} | "
            f"{len(self.samples):4d} samples | "
            f"frames: {FRAMES_ROOT / split}"
        )

    # ------------------------------------------------------------------
    def _check_paths(self):
        for p, label in [(FRAMES_ROOT, "FRAMES_ROOT"), (ANNOT_ROOT, "ANNOT_ROOT")]:
            if not p.exists():
                raise FileNotFoundError(
                    f"{label} not found: {p}\n"
                    f"Check that BASE_PATH = {BASE_PATH} is correct."
                )
        split_dir = FRAMES_ROOT / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Frame split directory not found: {split_dir}"
            )

    # ------------------------------------------------------------------
    def _load_annotations(self) -> List[Dict]:
        """
        Parse PHOENIX-2014-T.<split>.corpus.csv.

        Each row becomes a sample dict with keys:
          video_id, frame_dir, gloss (list of str), translation (list of str)
        """
        csv_path = ANNOT_ROOT / f"PHOENIX-2014-T.{self.split}.corpus.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

        samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                video_id     = row["name"].strip()
                gloss_tokens = row["orth"].strip().split()
                # Wrap translation with BOS / EOS for the decoder
                trans_tokens = (
                    [self.text_vocab.BOS]
                    + row["translation"].strip().split()
                    + [self.text_vocab.EOS]
                )
                # Frames live in: FRAMES_ROOT/<split>/<video_id>/*.png
                frame_dir = FRAMES_ROOT / self.split / video_id

                samples.append({
                    "video_id":    video_id,
                    "frame_dir":   frame_dir,
                    "gloss":       gloss_tokens,
                    "translation": trans_tokens,
                })
        return samples

    # ------------------------------------------------------------------
    def _load_frames(self, frame_dir: Path) -> Tuple[torch.Tensor, int]:
        """
        Load all frames from frame_dir, apply temporal stride,
        pad/truncate to self.max_frames.

        Returns:
            frames   : (max_frames, C, H, W)
            T_actual : number of real (non-padded) frames
        """
        frame_files = sorted(frame_dir.glob("*.png"))
        if not frame_files:
            frame_files = sorted(frame_dir.glob("*.jpg"))
        if not frame_files:
            raise FileNotFoundError(f"No frame images found in: {frame_dir}")

        # Temporal subsampling
        frame_files = frame_files[:: self.temporal_stride]

        frames = []
        for fp in frame_files:
            img = Image.open(fp).convert("RGB")
            if self.transform:
                img = self.transform(img)      # -> tensor (C, H, W)
            frames.append(img)

        frames   = torch.stack(frames)         # (T, C, H, W)
        T_actual = frames.shape[0]

        # Truncate
        if T_actual > self.max_frames:
            frames   = frames[: self.max_frames]
            T_actual = self.max_frames

        # Zero-pad
        if T_actual < self.max_frames:
            pad    = torch.zeros(
                self.max_frames - T_actual, *frames.shape[1:],
                dtype=frames.dtype
            )
            frames = torch.cat([frames, pad], dim=0)

        return frames, T_actual

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        frames, frame_len = self._load_frames(sample["frame_dir"])  # (T, C, H, W)

        gloss_ids = torch.tensor(
            self.gloss_vocab.encode(sample["gloss"]), dtype=torch.long
        )

        item = {
            "video_id":  sample["video_id"],
            "frames":    frames,                                  # (max_frames, C, H, W)
            "frame_len": torch.tensor(frame_len, dtype=torch.long),
            "gloss":     gloss_ids,                              # (G,)
            "gloss_len": torch.tensor(len(sample["gloss"]), dtype=torch.long),
        }

        if self.return_translation:
            trans_ids = torch.tensor(
                self.text_vocab.encode(sample["translation"]), dtype=torch.long
            )
            item["translation"]     = trans_ids                  # (S,)
            item["translation_len"] = torch.tensor(
                len(sample["translation"]), dtype=torch.long
            )

        return item


# ──────────────────────────────────────────────────────────────────────────────
# Collate function
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collates a list of sample dicts into batched tensors.
    Frames are already padded to max_frames, so they stack directly.
    Gloss and translation sequences are padded to the longest in the batch.
    """
    video_ids  = [b["video_id"]  for b in batch]
    frames     = torch.stack([b["frames"]    for b in batch])   # (B, T, C, H, W)
    frame_lens = torch.stack([b["frame_len"] for b in batch])   # (B,)

    # Pad gloss sequences
    max_gloss  = max(b["gloss_len"].item() for b in batch)
    gloss_pad  = torch.zeros(len(batch), max_gloss, dtype=torch.long)
    gloss_lens = []
    for i, b in enumerate(batch):
        g = b["gloss"]
        gloss_pad[i, : len(g)] = g
        gloss_lens.append(b["gloss_len"])

    out = {
        "video_ids":  video_ids,
        "frames":     frames,
        "frame_lens": frame_lens,
        "gloss":      gloss_pad,
        "gloss_lens": torch.stack(gloss_lens),
    }

    # Pad translation sequences (SLT only)
    if "translation" in batch[0]:
        max_trans  = max(b["translation_len"].item() for b in batch)
        trans_pad  = torch.zeros(len(batch), max_trans, dtype=torch.long)
        trans_lens = []
        for i, b in enumerate(batch):
            t = b["translation"]
            trans_pad[i, : len(t)] = t
            trans_lens.append(b["translation_len"])
        out["translation"]      = trans_pad
        out["translation_lens"] = torch.stack(trans_lens)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary builder  (call once before training)
# ──────────────────────────────────────────────────────────────────────────────

def build_vocabularies() -> Tuple[Vocabulary, Vocabulary]:
    """
    Build gloss and German text vocabularies from the train split only
    (as required by the PHOENIX-2014-T benchmark protocol).

    Uses ANNOT_ROOT directly -- no arguments needed.

    Returns:
        gloss_vocab : Vocabulary for sign gloss tokens
        text_vocab  : Vocabulary for German translation tokens
    """
    csv_path = ANNOT_ROOT / "PHOENIX-2014-T.train.corpus.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {csv_path}")

    gloss_vocab = Vocabulary()
    text_vocab  = Vocabulary()
    all_gloss, all_trans = [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            all_gloss.append(row["orth"].strip().split())
            all_trans.append(row["translation"].strip().split())

    gloss_vocab.build_from_sequences(all_gloss)
    text_vocab.build_from_sequences(all_trans)

    print(f"[Vocab] Gloss vocab size : {len(gloss_vocab):5d}  (from {csv_path.name})")
    print(f"[Vocab] Text  vocab size : {len(text_vocab):5d}  (from {csv_path.name})")
    return gloss_vocab, text_vocab