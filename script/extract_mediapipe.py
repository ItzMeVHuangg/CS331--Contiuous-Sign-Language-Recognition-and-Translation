"""
scripts/extract_mediapipe.py
─────────────────────────────
Pre-extract MediaPipe holistic landmarks from PHOENIX-2014-T frames.
Must be run ONCE before training Mediapipe variants (E / F).

Output
──────
For each video, saves one  .npy  file:
    <mediapipe_kpts_root>/<split>/<video_id>.npy
    shape : (T, 225)   float32
    layout: right_hand (63) | left_hand (63) | pose (99)
            = 21×3 + 21×3 + 33×3

All coordinates are normalised to [0, 1] in frame-pixel space.
Missing landmarks (hand not visible) are filled with zeros.

Usage
─────
    python scripts/extract_mediapipe.py \\
        --frames_root /path/to/fullFrame-210x260px \\
        --out_root    /path/to/keypoints \\
        --splits train dev test \\
        --num_workers 8

Requirements
────────────
    pip install mediapipe opencv-python tqdm
"""

import argparse
import csv
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
    import cv2
    _MP_OK = True
except ImportError:
    _MP_OK = False


ANNOT_ROOT = Path("/datastore/cndt_phungdtm/KLTN_HoangBinh/Dataset/"
                  "PHOENIX-2014-T/annotations/manual")

LANDMARK_DIM = 225   # right_hand(63) + left_hand(63) + pose(99)


def _extract_video(args):
    """Worker function: extract landmarks for one video."""
    video_id, frame_dir, out_path = args

    if out_path.exists():
        return video_id, True, "cached"

    mp_holistic = mp.solutions.holistic
    holistic    = mp_holistic.Holistic(
        static_image_mode        = True,
        model_complexity         = 1,
        min_detection_confidence = 0.5,
    )

    frame_files = sorted(frame_dir.glob("*.png"))
    if not frame_files:
        frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        return video_id, False, "no frames"

    all_kpts = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is None:
            all_kpts.append(np.zeros(LANDMARK_DIM, dtype=np.float32))
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        row = np.zeros(LANDMARK_DIM, dtype=np.float32)

        # Right hand (21 × 3 = 63, offset=0)
        if results.right_hand_landmarks:
            for j, lm in enumerate(results.right_hand_landmarks.landmark):
                row[j*3 : j*3+3] = [lm.x, lm.y, lm.z]

        # Left hand (21 × 3 = 63, offset=63)
        if results.left_hand_landmarks:
            for j, lm in enumerate(results.left_hand_landmarks.landmark):
                row[63 + j*3 : 63 + j*3 + 3] = [lm.x, lm.y, lm.z]

        # Pose (33 × 3 = 99, offset=126; skip visibility)
        if results.pose_landmarks:
            for j, lm in enumerate(results.pose_landmarks.landmark):
                row[126 + j*3 : 126 + j*3 + 3] = [lm.x, lm.y, lm.z]

        all_kpts.append(row)

    holistic.close()

    kpts_array = np.stack(all_kpts, axis=0).astype(np.float32)  # (T, 225)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), kpts_array)

    return video_id, True, f"T={len(all_kpts)}"


def extract_split(
    split: str,
    frames_root: Path,
    out_root: Path,
    num_workers: int = 4,
):
    csv_path = ANNOT_ROOT / f"PHOENIX-2014-T.{split}.corpus.csv"
    if not csv_path.exists():
        print(f"[WARN] CSV not found: {csv_path}. Skipping split '{split}'.")
        return

    tasks = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            video_id  = row["name"].strip()
            frame_dir = frames_root / split / video_id
            out_path  = out_root / split / f"{video_id}.npy"
            tasks.append((video_id, frame_dir, out_path))

    print(f"\n[{split}] {len(tasks)} videos  →  {out_root / split}")

    ok, fail = 0, 0
    if num_workers <= 1:
        for args in tqdm(tasks, desc=split):
            vid, success, msg = _extract_video(args)
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  [FAIL] {vid}: {msg}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as exe:
            futures = {exe.submit(_extract_video, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks), desc=split):
                vid, success, msg = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    print(f"  [FAIL] {vid}: {msg}")

    print(f"[{split}] Done: {ok} OK, {fail} failed.")


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints")
    parser.add_argument("--frames_root", required=True,
                        help="Path to fullFrame-210x260px directory")
    parser.add_argument("--out_root",    required=True,
                        help="Output directory for .npy keypoint files")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if not _MP_OK:
        print("ERROR: mediapipe not installed.")
        print("  pip install mediapipe opencv-python")
        return

    frames_root = Path(args.frames_root)
    out_root    = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        extract_split(split, frames_root, out_root, args.num_workers)

    print("\nExtraction complete.")
    print(f"Keypoints saved to: {out_root}")
    print(f"Expected shape per video: (T, {LANDMARK_DIM})  float32")


if __name__ == "__main__":
    main()