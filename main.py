
import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CSLR + CSLT Pipeline")
    parser.add_argument(
        "--stage",
        choices=["cslr", "slt", "hpo_cslr", "hpo_slt", "eval"],
        required=True,
    )
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--cslr_ckpt", default="checkpoints/cslr/best_cslr.pth")
    parser.add_argument("--slt_ckpt",  default="checkpoints/slt/best_slt.pth")
    parser.add_argument("--n_trials",  type=int, default=50, help="Optuna trials")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Stage dispatch ────────────────────────────────────────────
    if args.stage == "cslr":
        from training.train_cslr import train_cslr
        train_cslr(cfg)

    elif args.stage == "slt":
        from training.train_slt import train_slt
        train_slt(cfg, args.cslr_ckpt)

    elif args.stage == "hpo_cslr":
        from training.hyperparameter_tuning import run_hpo
        best = run_hpo(cfg, "cslr", args.n_trials)
        print("Best CSLR params:", best)

    elif args.stage == "hpo_slt":
        from training.hyperparameter_tuning import run_hpo
        best = run_hpo(cfg, "slt", args.n_trials, cslr_ckpt=args.cslr_ckpt)
        print("Best SLT params:", best)

    elif args.stage == "eval":
        _run_evaluation(cfg, args.cslr_ckpt, args.slt_ckpt)


def _run_evaluation(cfg, cslr_ckpt_path, slt_ckpt_path):
    """Run full evaluation on the test split."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from data.dataset import PhoenixDataset, collate_fn, build_transforms
    from training.train_slt import CSLTModel
    from utils.metrics import compute_wer, compute_bleu, compute_rouge, compute_meteor
    from utils.ctc_decoder import batch_ctc_decode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    slt_ckpt   = torch.load(slt_ckpt_path, map_location="cpu", weights_only=False)
    gloss_vocab = slt_ckpt["gloss_vocab"]
    text_vocab  = slt_ckpt["text_vocab"]

    # Test dataset
    test_ds = PhoenixDataset(
        split="test", 
        
        gloss_vocab=gloss_vocab, text_vocab=text_vocab,
        max_frames=cfg["data"]["max_frames"],
        temporal_stride=cfg["data"]["temporal_stride"],
        transform=build_transforms("test", cfg["data"]["img_height"], cfg["data"]["img_width"]),
        return_translation=True,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["slt"]["batch_size"],
                              collate_fn=collate_fn,
                              num_workers=cfg["data"]["num_workers"])

    # Load model
    model = CSLTModel(cfg, len(gloss_vocab), len(text_vocab)).to(device)
    model.load_state_dict(slt_ckpt["model"])
    model.eval()

    bos_idx = text_vocab.token2idx[text_vocab.BOS]
    eos_idx = text_vocab.token2idx[text_vocab.EOS]

    hyp_glosses, ref_glosses = [], []
    hyp_sents,   ref_sents   = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test Eval]"):
            frames     = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            gloss      = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)
            tgt        = batch["translation"]

            # CSLR: greedy decode
            log_probs, _ = model.cslr(frames, frame_lens)
            T_out = log_probs.size(0)
            preds_ids = batch_ctc_decode(
                log_probs, frame_lens.clamp(max=T_out),
                blank_idx=cfg["cslr"]["ctc_blank_idx"]
            )
            for b, pred in enumerate(preds_ids):
                hyp_glosses.append(gloss_vocab.decode(pred))
                ref_glosses.append(gloss_vocab.decode(
                    gloss[b, :gloss_lens[b].item()].tolist(), skip_special=False
                ))

            # SLT: translate
            pred_ids = model.translate(
                frames, frame_lens, gloss, gloss_lens, bos_idx, eos_idx
            )
            for b in range(pred_ids.size(0)):
                hyp_sents.append(" ".join(text_vocab.decode(pred_ids[b].tolist())))
                ref_sents.append(" ".join(text_vocab.decode(tgt[b].tolist())))

    # Compute all metrics
    wer    = compute_wer(hyp_glosses, ref_glosses)
    bleu   = compute_bleu(hyp_sents, ref_sents)
    rouge  = compute_rouge(hyp_sents, ref_sents)
    meteor = compute_meteor(hyp_sents, ref_sents)

    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"  CSLR WER   : {wer*100:.2f}%")
    print(f"  BLEU-4     : {bleu['bleu']:.2f}")
    print(f"  BLEU-1/2/3 : {bleu['bleu1']:.2f} / {bleu['bleu2']:.2f} / {bleu['bleu3']:.2f}")
    print(f"  ROUGE-1    : {rouge['rouge1']:.4f}")
    print(f"  ROUGE-2    : {rouge['rouge2']:.4f}")
    print(f"  ROUGE-L    : {rouge['rougeL']:.4f}")
    print(f"  METEOR     : {meteor:.4f}")
    print("="*60)

    # Show example predictions
    print("\n[Sample Predictions]")
    for i in range(min(3, len(hyp_sents))):
        print(f"  REF : {ref_sents[i]}")
        print(f"  HYP : {hyp_sents[i]}")
        print()


if __name__ == "__main__":
    main()