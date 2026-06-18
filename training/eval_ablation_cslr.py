import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from data.dataset import build_vocabularies
from training.train_ablation import make_loader, build_cslr_model, CSLR_VARIANTS
from utils.ctc_decoder import batch_ctc_decode
from utils.metrics import compute_wer


def evaluate_variant_cslr(
    cfg_path: str,
    variant: str,
    ckpt_path: str,
    split: str = "test",
    decode_mode: str = "greedy",
    beam_width: int = 10,
):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    variant = variant.upper()
    if variant not in CSLR_VARIANTS:
        raise ValueError(f"Unknown CSLR variant '{variant}'. Available: {list(CSLR_VARIANTS.keys())}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' state_dict: {ckpt_path}")

    variant_cfg = CSLR_VARIANTS[variant]

    gloss_vocab, text_vocab = build_vocabularies()
    loader = make_loader(
        split=split,
        return_translation=False,
        cfg=cfg,
        gloss_vocab=gloss_vocab,
        text_vocab=text_vocab,
        encoder_type=variant_cfg["encoder"],
        seed=cfg.get("seed", 42),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cslr_model(
        cfg,
        num_classes=len(gloss_vocab),
        encoder_type=variant_cfg["encoder"],
        seq_model_type=variant_cfg["seq_model"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    hyp_glosses, ref_glosses = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[CSLR-{variant} {split} eval]"):
            frames = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)

            log_probs, _, adj_lens = model(frames, frame_lens)
            preds = batch_ctc_decode(
                log_probs,
                adj_lens,
                blank_idx=cfg["cslr"]["ctc_blank_idx"],
                mode=decode_mode,
                beam_width=beam_width,
            )

            for b, pred in enumerate(preds):
                hyp_glosses.append(gloss_vocab.decode(pred))
                ref_glosses.append(
                    gloss_vocab.decode(
                        gloss[b, :gloss_lens[b].item()].tolist(), skip_special=False
                    )
                )

    wer = compute_wer(hyp_glosses, ref_glosses)

    print("\n" + "=" * 64)
    print(f"CSLR Variant {variant} | Split: {split}")
    print(f"Checkpoint: {Path(ckpt_path).resolve()}")
    print(f"CTC decode: {decode_mode} (beam_width={beam_width})")
    print(f"WER: {wer * 100:.2f}%")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ablation CSLR variant on split")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--variant", required=True, help="One of A..I (CSLR variants only)")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--decode", default="greedy", choices=["greedy", "beam"],
                        help="CTC decoding mode")
    parser.add_argument("--beam_width", type=int, default=10,
                        help="Beam width used when --decode beam")
    args = parser.parse_args()

    evaluate_variant_cslr(
        args.config,
        args.variant,
        args.ckpt,
        args.split,
        decode_mode=args.decode,
        beam_width=args.beam_width,
    )
