
import torch
import torch.nn.functional as F
from typing import List, Tuple
import heapq


def ctc_greedy_decode(
    log_probs: torch.Tensor,   # (T, B, C) or (T, C) for single sample
    blank_idx: int = 0,
) -> List[List[int]]:
 
    # Handle (T, C) single sample
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(1)   # (T, 1, C)

    T, B, C = log_probs.shape
    best_path = log_probs.argmax(dim=-1)     # (T, B)
    best_path = best_path.permute(1, 0)      # (B, T)

    results = []
    for b in range(B):
        path = best_path[b].tolist()
        decoded = []
        prev = None
        for token in path:
            if token != blank_idx and token != prev:
                decoded.append(token)
            prev = token
        results.append(decoded)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Simple prefix beam search
# ──────────────────────────────────────────────────────────────────────────────

def ctc_beam_decode(
    log_probs: torch.Tensor,   # (T, C) — single sample (no batch)
    beam_width: int = 10,
    blank_idx: int = 0,
) -> List[int]:

    T, C = log_probs.shape
    probs = log_probs.exp().cpu().numpy()  # convert to probabilities

    # Beam: list of (prefix_tuple, (prob_blank, prob_nonblank))
    # We track log-probs to avoid underflow
    NEG_INF = float("-inf")

    # Initialize
    beams = {(): (0.0, NEG_INF)}  # empty prefix: (log p_blank, log p_non_blank)

    import math

    def log_sum_exp(a, b):
        if a == NEG_INF:
            return b
        if b == NEG_INF:
            return a
        return max(a, b) + math.log1p(math.exp(-abs(a - b)))

    for t in range(T):
        new_beams = {}

        # Extend each beam
        for prefix, (pb, pnb) in beams.items():
            p_total = log_sum_exp(pb, pnb)

            for c in range(C):
                lp = math.log(probs[t, c] + 1e-30)

                if c == blank_idx:
                    # Extend with blank
                    key = prefix
                    pb_new, pnb_new = new_beams.get(key, (NEG_INF, NEG_INF))
                    new_beams[key] = (log_sum_exp(pb_new, p_total + lp), pnb_new)
                else:
                    # Extend with non-blank
                    key = prefix + (c,)
                    pb_new, pnb_new = new_beams.get(key, (NEG_INF, NEG_INF))

                    if len(prefix) > 0 and prefix[-1] == c:
                        # Same as last token: can only extend from blank path
                        new_beams[key] = (pb_new, log_sum_exp(pnb_new, pb + lp))
                    else:
                        new_beams[key] = (pb_new, log_sum_exp(pnb_new, p_total + lp))

        # Prune to beam_width
        def beam_score(item):
            pb, pnb = item[1]
            return log_sum_exp(pb, pnb)

        beams = dict(
            sorted(new_beams.items(), key=beam_score, reverse=True)[:beam_width]
        )

    # Return best prefix
    best = max(beams.items(), key=lambda x: log_sum_exp(x[1][0], x[1][1]))
    return list(best[0])


def batch_ctc_decode(
    log_probs: torch.Tensor,   # (T, B, C)
    lengths:   torch.Tensor,   # (B,) actual T per sample
    blank_idx: int = 0,
    mode: str = "greedy",      # "greedy" | "beam"
    beam_width: int = 10,
) -> List[List[int]]:
    """Decode a full batch."""
    T, B, C = log_probs.shape
    results = []
    for b in range(B):
        actual_len = lengths[b].item()
        lp = log_probs[:actual_len, b, :]   # (actual_T, C)
        if mode == "beam":
            pred = ctc_beam_decode(lp, beam_width=beam_width, blank_idx=blank_idx)
        else:
            pred = ctc_greedy_decode(lp.unsqueeze(1), blank_idx=blank_idx)[0]
        results.append(pred)
    return results