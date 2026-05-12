

import sys
import copy
import argparse
import yaml
import json
from pathlib import Path

import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners  import HyperbandPruner, MedianPruner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# CSLR Objective
# ──────────────────────────────────────────────────────────────────────────────

def cslr_objective(trial: optuna.Trial, base_cfg: dict) -> float:
    """Optuna objective: returns WER (to minimize)."""
    from training.train_cslr import train_cslr

    cfg = copy.deepcopy(base_cfg)
    hpo = cfg["hpo"]["search_space"]

    # ── Sample hyperparameters ────────────────────────────────────
    trial_params = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            hpo["learning_rate"][0], hpo["learning_rate"][1], log=True
        ),
        "hidden_size": trial.suggest_categorical(
            "hidden_size", hpo["hidden_size"]
        ),
        "num_layers": trial.suggest_categorical(
            "num_layers", hpo["num_layers"]
        ),
        "dropout": trial.suggest_float(
            "dropout", hpo["dropout"][0], hpo["dropout"][1]
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", hpo["batch_size"]
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            hpo["weight_decay"][0], hpo["weight_decay"][1], log=True
        ),
    }

    # Reduce epochs for HPO runs (use 1/4 of full training)
    cfg["cslr"]["num_epochs"] = max(5, cfg["cslr"]["num_epochs"] // 4)

    print(f"\n[Trial {trial.number}] Params: {trial_params}")

    try:
        wer = train_cslr(cfg, trial_params=trial_params)
    except Exception as e:
        print(f"[Trial {trial.number}] FAILED: {e}")
        raise optuna.exceptions.TrialPruned()

    print(f"[Trial {trial.number}] WER = {wer*100:.2f}%")
    return wer


# ──────────────────────────────────────────────────────────────────────────────
# SLT Objective
# ──────────────────────────────────────────────────────────────────────────────

def slt_objective(trial: optuna.Trial, base_cfg: dict, cslr_ckpt: str) -> float:
    """Optuna objective for SLT: returns -BLEU (to minimize = maximize BLEU)."""
    from training.train_slt import train_slt

    cfg = copy.deepcopy(base_cfg)

    # Sample SLT-specific hyperparameters
    cfg["slt"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-5, 5e-4, log=True
    )
    cfg["slt"]["d_model"] = trial.suggest_categorical(
        "d_model", [256, 512]
    )
    cfg["slt"]["nhead"] = trial.suggest_categorical(
        "nhead", [4, 8]
    )
    cfg["slt"]["num_encoder_layers"] = trial.suggest_int(
        "num_encoder_layers", 1, 4
    )
    cfg["slt"]["num_decoder_layers"] = trial.suggest_int(
        "num_decoder_layers", 1, 4
    )
    cfg["slt"]["dropout"] = trial.suggest_float(
        "dropout", 0.05, 0.3
    )
    cfg["slt"]["label_smoothing"] = trial.suggest_float(
        "label_smoothing", 0.0, 0.2
    )
    cfg["fusion"]["mode"] = trial.suggest_categorical(
        "fusion_mode", ["concat", "add", "attention"]
    )

    # Short run for HPO
    cfg["slt"]["num_epochs"] = max(5, cfg["slt"]["num_epochs"] // 4)

    try:
        bleu = train_slt(cfg, cslr_ckpt)
    except Exception as e:
        print(f"[Trial {trial.number}] FAILED: {e}")
        raise optuna.exceptions.TrialPruned()

    return -bleu   # minimize negative BLEU


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_hpo(cfg: dict, stage: str, n_trials: int, cslr_ckpt: str = None):
    """Launch Optuna study."""

    # Sampler
    sampler_name = cfg["hpo"].get("sampler", "tpe")
    if sampler_name == "tpe":
        sampler = TPESampler(seed=42)
    elif sampler_name == "random":
        sampler = RandomSampler(seed=42)
    elif sampler_name == "cmaes":
        sampler = CmaEsSampler(seed=42)
    else:
        sampler = TPESampler(seed=42)

    # Pruner
    pruner_name = cfg["hpo"].get("pruner", "hyperband")
    if pruner_name == "hyperband":
        pruner = HyperbandPruner()
    elif pruner_name == "median":
        pruner = MedianPruner()
    else:
        pruner = HyperbandPruner()

    study_name = f"cslr_cslt_{stage}"
    storage    = f"sqlite:///checkpoints/optuna_{stage}.db"
    Path("checkpoints").mkdir(exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    if stage == "cslr":
        study.optimize(
            lambda trial: cslr_objective(trial, cfg),
            n_trials=n_trials,
            timeout=cfg["hpo"].get("timeout", 36000),
            show_progress_bar=True,
        )
    else:
        assert cslr_ckpt is not None, "--cslr_ckpt required for SLT HPO"
        study.optimize(
            lambda trial: slt_objective(trial, cfg, cslr_ckpt),
            n_trials=n_trials,
            timeout=cfg["hpo"].get("timeout", 36000),
            show_progress_bar=True,
        )

    # ── Report results ────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"HPO COMPLETE  |  Stage: {stage}")
    print(f"Best trial:   #{study.best_trial.number}")
    print(f"Best value:   {study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*60)

    # Save best params to JSON
    results_path = Path("checkpoints") / f"best_hpo_{stage}.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_value":  study.best_value,
            "best_params": study.best_params,
        }, f, indent=2)
    print(f"Results saved to {results_path}")

    # Show importance plot if optuna-dashboard available
    try:
        import optuna.visualization as vis
        fig = vis.plot_param_importances(study)
        fig.write_html(f"checkpoints/hpo_{stage}_importance.html")
        print(f"Importance plot: checkpoints/hpo_{stage}_importance.html")
    except Exception:
        pass

    return study.best_params


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",     choices=["cslr", "slt"], required=True)
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--n_trials",  type=int, default=50)
    parser.add_argument("--cslr_ckpt", default="checkpoints/cslr/best_cslr.pth",
                        help="Required for SLT HPO")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_hpo(cfg, args.stage, args.n_trials, args.cslr_ckpt)