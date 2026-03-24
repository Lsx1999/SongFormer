"""Download SongFormBench dataset from HuggingFace and prepare GT annotations."""

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm


LABEL_MAP = {
    "intro": "intro",
    "verse": "verse",
    "chorus": "chorus",
    "bridge": "bridge",
    "instrumental": "inst",
    "inst": "inst",
    "outro": "outro",
    "silence": "silence",
    "pre-chorus": "pre-chorus",
    "prechorus": "pre-chorus",
    "interlude": "inst",
}


def normalize_label(label: str) -> str:
    label = label.strip().lower()
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    return label


def segments_to_msa_txt(segments) -> str:
    """Convert HuggingFace dataset segments to MSA TXT format.

    segments: list of dicts with 'start', 'end', 'label' keys
    """
    lines = []
    for seg in segments:
        start = float(seg["start"])
        label = normalize_label(seg["label"])
        lines.append(f"{start:.6f} {label}")
    if segments:
        end = float(segments[-1]["end"])
        lines.append(f"{end:.6f} end")
    return "\n".join(lines)


def download_and_prepare(output_dir: str, use_mirror: bool = False):
    """Download SongFormBench and prepare audio + GT annotations.

    Directory structure created:
        output_dir/
            HarmonixSet/
                audio/          - audio files
                gt/             - GT annotations in MSA TXT format
                audio.scp       - list of audio paths for inference
            CN/
                audio/
                gt/
                audio.scp
    """
    repo_id = "ASLP-lab/SongFormBench"

    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    print(f"Loading dataset {repo_id} ...")
    ds = load_dataset(repo_id, trust_remote_code=True)

    for split_name in ds:
        split_ds = ds[split_name]
        print(f"\nProcessing split: {split_name} ({len(split_ds)} samples)")

        # Determine subset name from split
        if "harmonix" in split_name.lower() or "hx" in split_name.lower():
            subset = "HarmonixSet"
        elif "cn" in split_name.lower() or "chinese" in split_name.lower():
            subset = "CN"
        else:
            subset = split_name

        audio_dir = os.path.join(output_dir, subset, "audio")
        gt_dir = os.path.join(output_dir, subset, "gt")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        scp_lines = []

        for sample in tqdm(split_ds, desc=f"Preparing {subset}"):
            # Extract song ID
            song_id = sample.get("song_id") or sample.get("id") or sample.get("name")
            if song_id is None:
                continue
            song_id = str(song_id)

            # Save audio
            audio_info = sample.get("audio")
            if audio_info is not None:
                if isinstance(audio_info, dict) and "path" in audio_info:
                    src_path = audio_info["path"]
                    if src_path and os.path.exists(src_path):
                        ext = os.path.splitext(src_path)[1] or ".wav"
                        dst_path = os.path.join(audio_dir, f"{song_id}{ext}")
                        if not os.path.exists(dst_path):
                            os.symlink(os.path.abspath(src_path), dst_path)
                        scp_lines.append(os.path.abspath(dst_path))
                    elif "array" in audio_info:
                        import soundfile as sf
                        import numpy as np

                        dst_path = os.path.join(audio_dir, f"{song_id}.wav")
                        if not os.path.exists(dst_path):
                            sr = audio_info.get("sampling_rate", 24000)
                            sf.write(dst_path, np.array(audio_info["array"]), sr)
                        scp_lines.append(os.path.abspath(dst_path))
                elif isinstance(audio_info, str) and os.path.exists(audio_info):
                    ext = os.path.splitext(audio_info)[1] or ".wav"
                    dst_path = os.path.join(audio_dir, f"{song_id}{ext}")
                    if not os.path.exists(dst_path):
                        os.symlink(os.path.abspath(audio_info), dst_path)
                    scp_lines.append(os.path.abspath(dst_path))

            # Save GT annotation
            segments = sample.get("segments") or sample.get("labels") or sample.get("annotation")
            if segments is not None:
                gt_path = os.path.join(gt_dir, f"{song_id}.txt")
                if isinstance(segments, list) and len(segments) > 0:
                    if isinstance(segments[0], dict):
                        msa_txt = segments_to_msa_txt(segments)
                    elif isinstance(segments[0], (list, tuple)):
                        # Format: [[time, label], ...]
                        lines = []
                        for item in segments:
                            time_, label = float(item[0]), normalize_label(str(item[1]))
                            if label == "end":
                                lines.append(f"{time_:.6f} end")
                            else:
                                lines.append(f"{time_:.6f} {label}")
                        if not lines[-1].endswith("end"):
                            # Need to add end marker - use last segment's time
                            lines.append(f"{time_:.6f} end")
                        msa_txt = "\n".join(lines)
                    else:
                        continue
                    with open(gt_path, "w") as f:
                        f.write(msa_txt)
                elif isinstance(segments, str):
                    # Already in text format
                    with open(gt_path, "w") as f:
                        f.write(segments.strip())

        # Write SCP file
        scp_path = os.path.join(output_dir, subset, "audio.scp")
        with open(scp_path, "w") as f:
            f.write("\n".join(scp_lines) + "\n")
        print(f"  Saved {len(scp_lines)} audio paths to {scp_path}")
        print(f"  GT annotations saved to {gt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SongFormBench dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/SongFormBench",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--use_mirror",
        action="store_true",
        help="Use hf-mirror.com for Mainland China",
    )
    args = parser.parse_args()
    download_and_prepare(args.output_dir, args.use_mirror)
