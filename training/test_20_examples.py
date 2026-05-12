"""
test_20_examples.py
-------------------
Chạy inference 20 mẫu ngẫu nhiên từ PHOENIX-2014-T test split
cho cả CSLR (gloss) và SLT (text).

Cách dùng:
    python test_20_examples.py \
        --config  configs/config_resnet34.yaml \
        --cslr_ckpt  checkpoints/ablation/cslr_variant_A.pth \
        --slt_ckpt   checkpoints_final_resnet34/best_bleu4.pth \
        --n_samples  20 \
        --seed       42
"""

import argparse
import random
import time
from pathlib import Path

import torch
import numpy as np
import yaml
from jiwer import wer as compute_wer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def decode_ctc_greedy(log_probs: torch.Tensor, blank_id: int = 0) -> list[int]:
    """Greedy CTC decode — collapse repeats và bỏ blank."""
    ids = log_probs.argmax(-1).squeeze(0).tolist()
    out, prev = [], None
    for idx in ids:
        if idx != blank_id and idx != prev:
            out.append(idx)
        prev = idx
    return out


def ids_to_tokens(ids: list[int], vocab: dict[int, str]) -> list[str]:
    return [vocab.get(i, "<unk>") for i in ids]


def print_separator(title: str = ""):
    w = 70
    if title:
        pad = (w - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("─" * w)


# ──────────────────────────────────────────────
# Dataset sampling
# ──────────────────────────────────────────────

def sample_test_items(cfg: dict, n: int, seed: int):
    """
    Trả về danh sách n mẫu từ split test của PhoenixDataset.
    Thay thế PhoenixDataset bằng import thực tế của project bạn.
    """
    # ── THAY DÒNG NÀY bằng import đúng của project ──────────────
    from dataset.phoenix_dataset import PhoenixDataset  # noqa: PLC0415
    # ─────────────────────────────────────────────────────────────

    dataset = PhoenixDataset(
        split="test",
        frame_root=cfg["data"]["frame_root_test"],
        gloss_vocab_file=cfg["data"]["gloss_vocab"],
        text_vocab_file=cfg["data"]["text_vocab"],
        max_frames=cfg["data"].get("max_frames", 300),
    )

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))
    return dataset, indices


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_cslr(cfg: dict, ckpt_path: str, device: torch.device):
    """
    Tải CSLR model (ResNet34-2D + BiLSTM).
    Thay thế bằng import thực tế của project bạn.
    """
    # ── THAY bằng import đúng ────────────────────────────────────
    from models.cslr_model import CSLRModel  # noqa: PLC0415
    # ─────────────────────────────────────────────────────────────

    model = CSLRModel(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def load_slt(cfg: dict, ckpt_path: str, device: torch.device):
    """
    Tải SLT model (LateFusion + Transformer decoder).
    Thay thế bằng import thực tế của project bạn.
    """
    # ── THAY bằng import đúng ────────────────────────────────────
    from models.slt_model import SLTModel  # noqa: PLC0415
    # ─────────────────────────────────────────────────────────────

    model = SLTModel(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

@torch.no_grad()
def run_cslr_inference(model, frames: torch.Tensor, gloss_vocab_inv: dict,
                       device: torch.device, blank_id: int = 0) -> str:
    """Chạy CSLR và trả về chuỗi gloss dự đoán."""
    frames = frames.unsqueeze(0).to(device)          # (1, T, C, H, W)
    log_probs = model(frames)                        # (1, T', |vocab|)
    ids = decode_ctc_greedy(log_probs, blank_id)
    return " ".join(ids_to_tokens(ids, gloss_vocab_inv))


@torch.no_grad()
def run_slt_inference(model, frames: torch.Tensor, device: torch.device,
                      beam_size: int = 5, max_len: int = 50) -> str:
    """Chạy SLT với beam search và trả về câu dự đoán."""
    frames = frames.unsqueeze(0).to(device)
    # model.translate() — điều chỉnh tên hàm nếu khác trong project
    tokens = model.translate(frames, beam_size=beam_size, max_len=max_len)
    return " ".join(tokens)


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────

def compute_bleu4(refs: list[str], hyps: list[str]) -> float:
    ref_tokens  = [[r.split()] for r in refs]
    hyp_tokens  = [h.split() for h in hyps]
    smooth = SmoothingFunction().method1
    return corpus_bleu(ref_tokens, hyp_tokens,
                       weights=(0.25, 0.25, 0.25, 0.25),
                       smoothing_function=smooth) * 100


def compute_rouge_l(refs: list[str], hyps: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    return float(np.mean(scores))


# ──────────────────────────────────────────────
# Main test loop
# ──────────────────────────────────────────────

def test_20_examples(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    cfg = load_config(args.config)

    # ── Load vocab (inverse mapping id→token) ──
    # Điều chỉnh nếu vocab được load theo cách khác trong project
    from dataset.vocabulary import Vocabulary  # noqa: PLC0415
    gloss_vocab = Vocabulary(cfg["data"]["gloss_vocab"])
    text_vocab  = Vocabulary(cfg["data"]["text_vocab"])
    gloss_vocab_inv = {v: k for k, v in gloss_vocab.token2id.items()}
    text_vocab_inv  = {v: k for k, v in text_vocab.token2id.items()}

    # ── Load models ──
    print("Loading CSLR …")
    cslr_model = load_cslr(cfg, args.cslr_ckpt, device)

    print("Loading SLT  …")
    slt_model = load_slt(cfg, args.slt_ckpt, device)

    # ── Sample data ──
    print(f"Sampling {args.n_samples} examples from test split …\n")
    dataset, indices = sample_test_items(cfg, args.n_samples, args.seed)

    # ── Storage for aggregated metrics ──
    cslr_hyps, cslr_refs = [], []
    slt_hyps,  slt_refs  = [], []

    print_separator("PER-SAMPLE RESULTS")

    for rank, idx in enumerate(indices, 1):
        sample = dataset[idx]
        frames      = sample["frames"]           # Tensor (T, C, H, W)
        gloss_ref   = sample["gloss_text"]       # str
        text_ref    = sample["text"]             # str
        video_id    = sample.get("name", str(idx))

        t0 = time.time()

        # ── CSLR ──
        gloss_pred = run_cslr_inference(
            cslr_model, frames, gloss_vocab_inv, device
        )

        # ── SLT ──
        slt_pred = run_slt_inference(
            slt_model, frames, device,
            beam_size=args.beam_size
        )

        elapsed = time.time() - t0

        cslr_hyps.append(gloss_pred)
        cslr_refs.append(gloss_ref)
        slt_hyps.append(slt_pred)
        slt_refs.append(text_ref)

        # Pretty print
        print(f"\n[{rank:02d}/{args.n_samples}] {video_id}  ({elapsed:.1f}s)")
        print(f"  CSLR ref  : {gloss_ref}")
        print(f"  CSLR pred : {gloss_pred}")
        print(f"  SLT  ref  : {text_ref}")
        print(f"  SLT  pred : {slt_pred}")

    # ──────────────────────────────────────────
    # Aggregate metrics
    # ──────────────────────────────────────────
    print_separator("AGGREGATE METRICS (20 samples)")

    # CSLR — WER
    cslr_wer = compute_wer(cslr_refs, cslr_hyps) * 100
    print(f"\nCSLR  WER     : {cslr_wer:.2f}%")

    # SLT — BLEU-4 / ROUGE-L
    slt_bleu4   = compute_bleu4(slt_refs, slt_hyps)
    slt_rouge_l = compute_rouge_l(slt_refs, slt_hyps)
    print(f"SLT   BLEU-4  : {slt_bleu4:.2f}")
    print(f"SLT   ROUGE-L : {slt_rouge_l:.4f}")

    print_separator()

    # ──────────────────────────────────────────
    # Save results to TSV
    # ──────────────────────────────────────────
    out_path = Path(args.output) if args.output else Path("test_results_20.tsv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("idx\tvideo_id\tcslr_ref\tcslr_pred\tslt_ref\tslt_pred\n")
        for rank, idx in enumerate(indices):
            sample   = dataset[idx]
            video_id = sample.get("name", str(idx))
            f.write(
                f"{rank+1}\t{video_id}\t"
                f"{cslr_refs[rank]}\t{cslr_hyps[rank]}\t"
                f"{slt_refs[rank]}\t{slt_hyps[rank]}\n"
            )
    print(f"\nResults saved → {out_path}")

    return {
        "cslr_wer":   cslr_wer,
        "slt_bleu4":  slt_bleu4,
        "slt_rouge_l": slt_rouge_l,
    }


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 20 samples — CSLR & SLT")
    parser.add_argument("--config",    required=True,
                        help="Path to config YAML")
    parser.add_argument("--cslr_ckpt", required=True,
                        help="Path to CSLR checkpoint (.pth)")
    parser.add_argument("--slt_ckpt",  required=True,
                        help="Path to best SLT checkpoint (.pth)")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Số mẫu test (default: 20)")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam search width cho SLT (default: 5)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed để reproducible (default: 42)")
    parser.add_argument("--output",    default=None,
                        help="Đường dẫn file TSV kết quả (optional)")
    args = parser.parse_args()

    test_20_examples(args)
