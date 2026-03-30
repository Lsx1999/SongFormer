"""Summarize evaluation results from multiple subsets into a single MD file."""

import argparse
import json
import os

import pandas as pd


DISPLAY_METRICS = [
    ("num_samples", "Samples", "d"),
    ("acc", "ACC", ".4f"),
    ("HR.5F", "HR.5F", ".4f"),
    ("HR3F", "HR3F", ".4f"),
    ("HR1F", "HR1F", ".4f"),
    ("PWF", "PWF", ".4f"),
    ("Sf", "Sf", ".4f"),
    ("iou", "IoU", ".4f"),
]


def load_summary_csv(csv_path: str) -> dict:
    """Load a single eval_infer_summary.csv and return as dict."""
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {}
    row = df.iloc[0].to_dict()
    return row


def format_metric_table(name: str, metrics: dict) -> str:
    """Format a single subset's metrics as a markdown table."""
    lines = [f"## {name}", ""]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    for key, display_name, fmt in DISPLAY_METRICS:
        if key in metrics:
            val = metrics[key]
            if fmt == "d":
                lines.append(f"| {display_name} | {int(val)} |")
            else:
                lines.append(f"| {display_name} | {val:{fmt}} |")

    # Add per-class IoU if present
    iou_keys = sorted([k for k in metrics if k.startswith("iou_")])
    if iou_keys:
        lines.append("")
        lines.append("### Per-class IoU")
        lines.append("")
        lines.append("| Label | IoU |")
        lines.append("|-------|-----|")
        for k in iou_keys:
            label = k.replace("iou_", "")
            lines.append(f"| {label} | {metrics[k]:.4f} |")

    lines.append("")
    return "\n".join(lines)


def format_rtf_table(rtf_files: dict[str, str]) -> str:
    """Format RTF statistics from JSON files into a markdown section."""
    lines = ["## RTF (Real-Time Factor)", ""]
    lines.append("| Subset | Samples | Audio Duration | Avg RTF | Min RTF | Max RTF | Std RTF |")
    lines.append("|--------|---------|----------------|---------|---------|---------|---------|")

    total_files = 0
    total_duration = 0.0
    weighted_rtf_sum = 0.0

    for subset, json_path in sorted(rtf_files.items()):
        if not os.path.isfile(json_path):
            print(f"Warning: RTF file {json_path} not found, skipping")
            continue
        with open(json_path) as f:
            data = json.load(f)
        n = data.get("num_files", 0)
        dur = data.get("total_audio_duration_s", 0.0)
        avg = data.get("avg_rtf", 0.0)
        lines.append(
            f"| {subset} | {n} | {dur:.1f}s ({dur/60:.1f}min) "
            f"| {avg:.4f} | {data.get('min_rtf', 0):.4f} "
            f"| {data.get('max_rtf', 0):.4f} | {data.get('std_rtf', 0):.4f} |"
        )
        total_files += n
        total_duration += dur
        weighted_rtf_sum += avg * dur

    if total_duration > 0:
        overall_avg_rtf = weighted_rtf_sum / total_duration
        lines.append(
            f"| **Overall** | {total_files} | {total_duration:.1f}s ({total_duration/60:.1f}min) "
            f"| **{overall_avg_rtf:.4f}** | - | - | - |"
        )

    lines.append("")
    return "\n".join(lines)


def summarize(eval_base_dir: str, output_path: str, subsets: list[str] | None = None,
              rtf_files: dict[str, str] | None = None):
    """Summarize evaluation results from multiple subsets.

    Args:
        eval_base_dir: Base directory containing per-subset eval output dirs.
            Expected structure:
                eval_base_dir/
                    HarmonixSet/
                        eval_infer_summary.csv
                    CN/
                        eval_infer_summary.csv
        output_path: Path for the output MD file.
        subsets: List of subset names to include. If None, auto-detect.
        rtf_files: Dict mapping subset name to RTF JSON file path.
    """
    if subsets is None:
        subsets = []
        for entry in sorted(os.listdir(eval_base_dir)):
            csv_path = os.path.join(eval_base_dir, entry, "eval_infer_summary.csv")
            if os.path.isfile(csv_path):
                subsets.append(entry)

    if not subsets:
        print(f"No evaluation results found in {eval_base_dir}")
        return

    md_parts = ["# SongFormer Evaluation Results", ""]

    for subset in subsets:
        csv_path = os.path.join(eval_base_dir, subset, "eval_infer_summary.csv")
        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found, skipping {subset}")
            continue

        metrics = load_summary_csv(csv_path)
        display_name = f"SongFormBench-{subset}"
        md_parts.append(format_metric_table(display_name, metrics))

    # Add RTF section
    if rtf_files:
        md_parts.append(format_rtf_table(rtf_files))

    md_content = "\n".join(md_parts)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md_content)

    print(f"Summary saved to {output_path}")
    print(md_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument(
        "--eval_base_dir",
        type=str,
        required=True,
        help="Base directory with per-subset eval results",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="eval_results/evaluation_summary.md",
        help="Output MD file path",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="*",
        default=None,
        help="Subset names to include (default: auto-detect)",
    )
    parser.add_argument(
        "--rtf_files",
        type=str,
        nargs="*",
        default=None,
        help="RTF JSON files in format SUBSET:path (e.g. HarmonixSet:/path/to/rtf.json)",
    )
    args = parser.parse_args()

    rtf_dict = None
    if args.rtf_files:
        rtf_dict = {}
        for item in args.rtf_files:
            subset_name, path = item.split(":", 1)
            rtf_dict[subset_name] = path

    summarize(args.eval_base_dir, args.output_path, args.subsets, rtf_dict)
