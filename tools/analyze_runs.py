#!/usr/bin/env python3
"""
Training Run Analyzer for PepTRM

Parses wandb run data and generates condensed summaries suitable for LLM analysis.
Outputs key metrics in a token-efficient format.

Usage:
    python tools/analyze_runs.py                    # List all runs
    python tools/analyze_runs.py --run RUNID        # Analyze specific run
    python tools/analyze_runs.py --latest           # Analyze latest run
    python tools/analyze_runs.py --recent N         # Analyze N most recent runs
    python tools/analyze_runs.py --compare A B      # Compare two runs
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


WANDB_DIR = Path("/home/jgehring/pepTRM/wandb")
LOCAL_LOGS_DIR = Path("/home/jgehring/pepTRM/logs")


def parse_run_id(run_dir: str) -> tuple[str, str, str]:
    """Extract date, time, and wandb ID from run directory name."""
    # Format: run-20251212_191603-iustpqgf
    match = re.match(r'run-(\d{8})_(\d{6})-(\w+)', run_dir)
    if match:
        date, time, wandb_id = match.groups()
        return date, time, wandb_id
    return "", "", ""


def list_runs(limit: int = 20) -> list[dict]:
    """List recent wandb runs with basic info."""
    runs = []
    for d in sorted(WANDB_DIR.iterdir(), reverse=True):
        if d.is_dir() and d.name.startswith("run-"):
            date, time, wandb_id = parse_run_id(d.name)
            summary_file = d / "files" / "wandb-summary.json"
            output_file = d / "files" / "output.log"

            run_info = {
                "dir": d.name,
                "wandb_id": wandb_id,
                "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                "time": f"{time[:2]}:{time[2:4]}:{time[4:]}",
                "has_summary": summary_file.exists(),
                "has_output": output_file.exists(),
            }

            # Try to get final metrics from summary
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                    run_info["final_step"] = summary.get("train/step", summary.get("_step", 0))
                    run_info["final_token_acc"] = summary.get("train/token_accuracy", 0)
                    run_info["final_seq_acc"] = summary.get("train/sequence_accuracy", 0)
                    run_info["val_easy_token"] = summary.get("val_easy/token_accuracy", 0)
                    run_info["val_hard_token"] = summary.get("val_hard/token_accuracy", 0)
                except:
                    pass

            runs.append(run_info)
            if len(runs) >= limit:
                break
    return runs


def load_summary(run_id: str) -> Optional[dict]:
    """Load wandb-summary.json for a run."""
    # Find matching run directory
    for d in WANDB_DIR.iterdir():
        if d.is_dir() and (run_id in d.name or d.name == f"run-{run_id}"):
            summary_file = d / "files" / "wandb-summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    return json.load(f)
    return None


def parse_output_log(run_id: str, sample_interval: int = 1000) -> list[dict]:
    """Parse output.log to extract training curve data."""
    # Find matching run directory
    run_dir = None
    for d in WANDB_DIR.iterdir():
        if d.is_dir() and (run_id in d.name or d.name == f"run-{run_id}"):
            run_dir = d
            break

    if not run_dir:
        return []

    output_file = run_dir / "files" / "output.log"
    if not output_file.exists():
        return []

    data_points = []
    with open(output_file) as f:
        for line in f:
            # Match training step lines
            # Format: Step 1000 | Loss: 2.8490 | Token Acc: 0.163 | Seq Acc: 0.000 | LR: 1.50e-04
            match = re.match(
                r'Step (\d+) \| Loss: ([\d.]+) \| Token Acc: ([\d.]+) \| Seq Acc: ([\d.]+)',
                line
            )
            if match:
                step = int(match.group(1))
                if step % sample_interval == 0:
                    data_points.append({
                        "step": step,
                        "loss": float(match.group(2)),
                        "token_acc": float(match.group(3)),
                        "seq_acc": float(match.group(4)),
                        "type": "train"
                    })

            # Match validation lines
            # Format: Val (Easy) | Token Acc: 0.280 | Seq Acc: 0.000
            val_match = re.match(
                r'Val \((\w+)\) \| Token Acc: ([\d.]+) \| Seq Acc: ([\d.]+)',
                line
            )
            if val_match:
                # Associate with previous training step
                val_set = val_match.group(1).lower()
                data_points.append({
                    "type": f"val_{val_set}",
                    "token_acc": float(val_match.group(2)),
                    "seq_acc": float(val_match.group(3)),
                })

    return data_points


def format_run_summary(summary: dict) -> str:
    """Format wandb summary into condensed, readable output."""
    lines = []

    # Header
    step = summary.get("train/step", summary.get("_step", 0))
    runtime = summary.get("_runtime", 0)
    runtime_hrs = runtime / 3600 if runtime else 0
    lines.append(f"=== Run Summary (Step {step}, {runtime_hrs:.1f}h) ===")
    lines.append("")

    # Training metrics
    lines.append("TRAINING:")
    lines.append(f"  Loss: {summary.get('train/loss', 0):.4f}")
    lines.append(f"  Token Acc: {summary.get('train/token_accuracy', 0)*100:.1f}%")
    lines.append(f"  Seq Acc: {summary.get('train/sequence_accuracy', 0)*100:.1f}%")

    # Mass metrics
    if "train/ppm_error" in summary:
        lines.append(f"  Mass Error: {summary.get('train/mass_error_da', 0):.1f} Da ({summary.get('train/ppm_error', 0):.0f} ppm)")
    lines.append("")

    # Per-step CE loss (CRITICAL for recursion analysis)
    lines.append("PER-STEP CE LOSS (recursion analysis):")
    ce_steps = []
    for i in range(8):
        key = f"train/ce_step_{i}"
        if key in summary:
            ce_steps.append((i, summary[key]))

    if ce_steps:
        # Show progression
        for i, loss in ce_steps:
            marker = ""
            if i == 0:
                marker = " (initial)"
            elif i == len(ce_steps) - 1:
                marker = " (final)"
            lines.append(f"  Step {i}: {loss:.4f}{marker}")

        # Calculate improvement
        if len(ce_steps) >= 2:
            initial = ce_steps[0][1]
            final = ce_steps[-1][1]
            improvement = (initial - final) / initial * 100
            lines.append(f"  Δ: {improvement:+.1f}% improvement (step 0→{len(ce_steps)-1})")

            # Check for plateau (all steps similar)
            losses = [l for _, l in ce_steps]
            if len(losses) > 2:
                mid_losses = losses[1:-1]
                if mid_losses:
                    variance = sum((l - sum(mid_losses)/len(mid_losses))**2 for l in mid_losses) / len(mid_losses)
                    if variance < 0.0001:
                        lines.append(f"  ⚠️  PLATEAU DETECTED: Steps 1-{len(ce_steps)-2} all ~{sum(mid_losses)/len(mid_losses):.4f}")
    lines.append("")

    # Recursion metrics (if present)
    recursion_keys = [k for k in summary.keys() if k.startswith("recursion/")]
    if recursion_keys:
        lines.append("RECURSION METRICS:")
        if "recursion/accuracy_gain" in summary:
            lines.append(f"  Accuracy Gain: {summary['recursion/accuracy_gain']*100:+.1f}%")
        if "recursion/total_edits" in summary:
            lines.append(f"  Total Edits: {summary['recursion/total_edits']*100:.1f}%")
        if "recursion/converged_by_step" in summary:
            lines.append(f"  Converged at Step: {summary['recursion/converged_by_step']}")

        # Edit rates per step
        edit_rates = [(k, summary[k]) for k in sorted(recursion_keys) if "edit_rate" in k]
        if edit_rates:
            lines.append("  Edit rates: " + " → ".join(f"{v*100:.1f}%" for k, v in edit_rates))
        lines.append("")

    # Validation metrics
    lines.append("VALIDATION:")
    for prefix, name in [("val_easy", "Easy"), ("val_hard", "Hard"),
                          ("val_proteometools", "ProteomeTools"), ("val_nine_species", "9-Species")]:
        token_key = f"{prefix}/token_accuracy"
        seq_key = f"{prefix}/sequence_accuracy"
        if token_key in summary:
            lines.append(f"  {name}: {summary[token_key]*100:.1f}% token, {summary.get(seq_key, 0)*100:.1f}% seq")
    lines.append("")

    # Curriculum info
    if "curriculum/stage_idx" in summary:
        lines.append("CURRICULUM:")
        lines.append(f"  Stage: {summary.get('curriculum/stage_idx', 0)}")
        lines.append(f"  Length: {summary.get('curriculum/min_length', 0)}-{summary.get('curriculum/max_length', 0)}")
        lines.append(f"  Dropout: {summary.get('curriculum/peak_dropout', 0)*100:.0f}%")
        if "curriculum/noise_peaks_low" in summary:
            lines.append(f"  Noise: {summary.get('curriculum/noise_peaks_low', 0)}-{summary.get('curriculum/noise_peaks_high', 0)} peaks")
        else:
            lines.append(f"  Noise: {summary.get('curriculum/noise_peaks', 0)} peaks")
        lines.append(f"  Clean ratio: {summary.get('curriculum/clean_data_ratio', 1.0)*100:.0f}%")
        lines.append(f"  Spectrum loss: {summary.get('curriculum/spectrum_loss_weight', 0)}")
        lines.append(f"  Precursor loss: {summary.get('curriculum/precursor_loss_weight', 0)}")

    return "\n".join(lines)


def format_run_list(runs: list[dict]) -> str:
    """Format list of runs as a table."""
    lines = ["RECENT RUNS:", ""]
    lines.append(f"{'ID':<12} {'Date':<12} {'Steps':>7} {'Train Tok':>10} {'Val Easy':>10} {'Val Hard':>10}")
    lines.append("-" * 75)

    for run in runs:
        lines.append(
            f"{run['wandb_id']:<12} "
            f"{run['date']:<12} "
            f"{run.get('final_step', 0):>7} "
            f"{run.get('final_token_acc', 0)*100:>9.1f}% "
            f"{run.get('val_easy_token', 0)*100:>9.1f}% "
            f"{run.get('val_hard_token', 0)*100:>9.1f}%"
        )

    return "\n".join(lines)


def format_training_curve(data_points: list[dict], every_n: int = 5000) -> str:
    """Format training curve as condensed ASCII."""
    lines = ["TRAINING CURVE:", ""]

    # Filter to training points only, at regular intervals
    train_points = [p for p in data_points if p["type"] == "train" and p["step"] % every_n == 0]

    if not train_points:
        return "No training data available"

    lines.append(f"{'Step':>7} {'Loss':>8} {'Token':>8} {'Seq':>8}")
    lines.append("-" * 35)

    for p in train_points:
        lines.append(
            f"{p['step']:>7} "
            f"{p['loss']:>8.4f} "
            f"{p['token_acc']*100:>7.1f}% "
            f"{p['seq_acc']*100:>7.1f}%"
        )

    return "\n".join(lines)


def compare_runs(run_a: str, run_b: str) -> str:
    """Compare two runs side by side."""
    summary_a = load_summary(run_a)
    summary_b = load_summary(run_b)

    if not summary_a or not summary_b:
        return "Could not load one or both runs"

    lines = [f"COMPARISON: {run_a} vs {run_b}", ""]

    metrics = [
        ("train/loss", "Loss", lambda x: f"{x:.4f}"),
        ("train/token_accuracy", "Token Acc", lambda x: f"{x*100:.1f}%"),
        ("train/sequence_accuracy", "Seq Acc", lambda x: f"{x*100:.1f}%"),
        ("val_easy/token_accuracy", "Val Easy", lambda x: f"{x*100:.1f}%"),
        ("val_hard/token_accuracy", "Val Hard", lambda x: f"{x*100:.1f}%"),
    ]

    lines.append(f"{'Metric':<15} {'Run A':>12} {'Run B':>12} {'Δ':>10}")
    lines.append("-" * 55)

    for key, name, fmt in metrics:
        val_a = summary_a.get(key, 0)
        val_b = summary_b.get(key, 0)
        if key == "train/loss":
            delta = val_a - val_b  # Lower is better
        else:
            delta = val_b - val_a  # Higher is better

        lines.append(
            f"{name:<15} {fmt(val_a):>12} {fmt(val_b):>12} {delta*100:>+9.1f}%"
        )

    # Compare per-step CE
    lines.append("")
    lines.append("Per-step CE comparison:")
    for i in range(8):
        key = f"train/ce_step_{i}"
        if key in summary_a and key in summary_b:
            va, vb = summary_a[key], summary_b[key]
            lines.append(f"  Step {i}: {va:.4f} vs {vb:.4f} (Δ {(va-vb)*100:+.1f}%)")

    return "\n".join(lines)


def get_latest_run() -> Optional[str]:
    """Get the ID of the latest run."""
    latest_link = WANDB_DIR / "latest-run"
    if latest_link.exists():
        target = latest_link.resolve().name
        _, _, wandb_id = parse_run_id(target)
        return wandb_id
    return None


def parse_local_log(log_file: Path) -> list[dict]:
    """Parse a local metrics JSONL file."""
    records = []
    if not log_file.exists():
        return records
    with open(log_file) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except:
                continue
    return records


def format_local_log_summary(log_file: Path, downsample: int = 5000) -> str:
    """Format local log file into condensed summary."""
    records = parse_local_log(log_file)
    if not records:
        return f"No data in {log_file}"

    lines = [f"=== LOCAL LOG: {log_file.name} ===", ""]

    # Separate training and validation records
    train_records = [r for r in records if r.get('type') != 'validation']
    val_records = [r for r in records if r.get('type') == 'validation']

    # Training curve (downsampled)
    lines.append("TRAINING CURVE:")
    lines.append(f"{'Step':>7} {'Loss':>7} {'TokAcc':>7} {'CE0':>6} {'CE7':>6} {'Δ':>6}")
    lines.append("-" * 50)

    for r in train_records:
        step = r.get('step', 0)
        if step % downsample == 0:
            loss = r.get('loss', 0)
            tok = r.get('token_accuracy', 0)
            ce0 = r.get('ce_step_0', 0)
            ce7 = r.get('ce_step_7', 0)
            delta = (ce0 - ce7) / ce0 * 100 if ce0 > 0 else 0
            lines.append(f"{step:>7} {loss:>7.3f} {tok*100:>6.1f}% {ce0:>6.3f} {ce7:>6.3f} {delta:>+5.1f}%")

    # Latest recursion metrics
    if train_records:
        latest = train_records[-1]
        lines.append("")
        lines.append("LATEST RECURSION METRICS:")
        edit_rates = []
        for i in range(1, 8):
            key = f'recursion/edit_rate_step_{i}'
            if key in latest:
                edit_rates.append(f"{latest[key]*100:.1f}%")
        if edit_rates:
            lines.append(f"  Edit rates: {' → '.join(edit_rates)}")

        acc_gain = latest.get('recursion/accuracy_gain', 0)
        lines.append(f"  Accuracy gain: {acc_gain*100:+.2f}%")

    # Validation summary
    if val_records:
        lines.append("")
        lines.append("VALIDATION HISTORY:")
        for r in val_records[-5:]:  # Last 5 validations
            step = r.get('step', 0)
            ve = r.get('val_easy_token_acc', 0) * 100
            vh = r.get('val_hard_token_acc', 0) * 100
            lines.append(f"  Step {step:>6}: Easy {ve:.1f}% | Hard {vh:.1f}%")

    return "\n".join(lines)


def list_local_logs() -> str:
    """List available local log files."""
    lines = ["LOCAL LOGS:", ""]
    if not LOCAL_LOGS_DIR.exists():
        return "No local logs directory found"

    log_files = sorted(LOCAL_LOGS_DIR.glob("metrics_*.jsonl"), reverse=True)
    if not log_files:
        return "No local log files found"

    for f in log_files[:10]:
        records = parse_local_log(f)
        train_records = [r for r in records if r.get('type') != 'validation']
        if train_records:
            max_step = max(r.get('step', 0) for r in train_records)
            lines.append(f"  {f.name}: {len(train_records)} records, max step {max_step}")
        else:
            lines.append(f"  {f.name}: empty")

    return "\n".join(lines)


def format_live_status() -> str:
    """Format live training status from latest-run output.log."""
    lines = ["LIVE TRAINING STATUS:", ""]

    latest_link = WANDB_DIR / "latest-run"
    if not latest_link.exists():
        return "No active run found"

    output_file = latest_link / "files" / "output.log"
    if not output_file.exists():
        return "Run found but no output.log yet"

    # Read last 100 lines
    with open(output_file) as f:
        all_lines = f.readlines()

    recent_lines = all_lines[-100:]

    # Extract latest metrics
    latest_train = None
    latest_val_easy = None
    latest_val_hard = None
    latest_val_pt = None
    latest_val_ns = None
    config_lines = []

    for line in recent_lines:
        line = line.strip()

        # Training step
        match = re.match(r'Step (\d+) \| Loss: ([\d.]+) \| Token Acc: ([\d.]+) \| Seq Acc: ([\d.]+)', line)
        if match:
            latest_train = {
                "step": int(match.group(1)),
                "loss": float(match.group(2)),
                "token_acc": float(match.group(3)),
                "seq_acc": float(match.group(4)),
            }

        # Validation
        val_match = re.match(r'Val \((\w+)\) \| Token Acc: ([\d.]+) \| Seq Acc: ([\d.]+)', line)
        if val_match:
            val_set = val_match.group(1)
            val_data = {
                "token_acc": float(val_match.group(2)),
                "seq_acc": float(val_match.group(3)),
            }
            if val_set == "Easy":
                latest_val_easy = val_data
            elif val_set == "Hard":
                latest_val_hard = val_data
            elif val_set == "ProteomeTools":
                latest_val_pt = val_data
            elif "Nine" in val_set or "Species" in val_set:
                latest_val_ns = val_data

    # Also get config from start of file
    for line in all_lines[:50]:
        if any(x in line for x in ['Batch size:', 'Max steps:', 'Stage', 'Device:']):
            config_lines.append(line.strip())

    if latest_train:
        lines.append(f"Current Step: {latest_train['step']:,}")
        lines.append(f"Train Loss: {latest_train['loss']:.4f}")
        lines.append(f"Train Token Acc: {latest_train['token_acc']*100:.1f}%")
        lines.append(f"Train Seq Acc: {latest_train['seq_acc']*100:.1f}%")
        lines.append("")

    if latest_val_easy:
        lines.append(f"Val (Easy): {latest_val_easy['token_acc']*100:.1f}% token, {latest_val_easy['seq_acc']*100:.1f}% seq")
    if latest_val_hard:
        lines.append(f"Val (Hard): {latest_val_hard['token_acc']*100:.1f}% token, {latest_val_hard['seq_acc']*100:.1f}% seq")
    if latest_val_pt:
        lines.append(f"Val (ProteomeTools): {latest_val_pt['token_acc']*100:.1f}% token, {latest_val_pt['seq_acc']*100:.1f}% seq")
    if latest_val_ns:
        lines.append(f"Val (Nine-Species): {latest_val_ns['token_acc']*100:.1f}% token, {latest_val_ns['seq_acc']*100:.1f}% seq")

    # Show curriculum stage based on step
    if latest_train:
        step = latest_train['step']
        # Based on AGGRESSIVE_NOISE_CURRICULUM stages
        stages = [
            (0, 5000, "Stage 0: Pure Foundation (100% clean)"),
            (5000, 15000, "Stage 1: Syntax & Grammar (80% clean, 10 noise)"),
            (15000, 30000, "Stage 2: Robustness (50% clean, 25+2 noise, 20% dropout)"),
            (30000, 50000, "Stage 3: Realistic Orbitrap (20% clean, 50+5 noise, 30% dropout)"),
            (50000, 105000, "Stage 4: Extreme Stress Test (0% clean, 150+10 noise, 50% dropout)"),
        ]
        for start, end, desc in stages:
            if start <= step < end:
                lines.append("")
                lines.append(f"Curriculum: {desc}")
                lines.append(f"Stage Progress: {step - start:,}/{end - start:,} ({(step - start)/(end - start)*100:.0f}%)")
                break

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze PepTRM training runs")
    parser.add_argument("--run", type=str, help="Analyze specific run by ID")
    parser.add_argument("--latest", action="store_true", help="Analyze latest run")
    parser.add_argument("--recent", type=int, help="Analyze N most recent runs")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two runs")
    parser.add_argument("--list", type=int, default=0, help="List N recent runs")
    parser.add_argument("--curve", action="store_true", help="Include training curve")
    parser.add_argument("--live", action="store_true", help="Show live training status")
    parser.add_argument("--local", type=str, nargs='?', const='latest', help="Analyze local log file (or 'latest')")
    parser.add_argument("--local-list", action="store_true", help="List local log files")

    args = parser.parse_args()

    if args.local_list:
        print(list_local_logs())
    elif args.local:
        if args.local == 'latest':
            log_files = sorted(LOCAL_LOGS_DIR.glob("metrics_*.jsonl"), reverse=True)
            if log_files:
                print(format_local_log_summary(log_files[0]))
            else:
                print("No local log files found")
        else:
            log_file = LOCAL_LOGS_DIR / args.local
            if not log_file.exists():
                log_file = Path(args.local)  # Try as absolute path
            print(format_local_log_summary(log_file))
    elif args.live:
        print(format_live_status())
    elif args.compare:
        print(compare_runs(args.compare[0], args.compare[1]))
    elif args.run:
        summary = load_summary(args.run)
        if summary:
            print(format_run_summary(summary))
            if args.curve:
                print("\n")
                data = parse_output_log(args.run)
                print(format_training_curve(data))
        else:
            print(f"Run {args.run} not found")
    elif args.latest:
        run_id = get_latest_run()
        if run_id:
            summary = load_summary(run_id)
            if summary:
                print(format_run_summary(summary))
                if args.curve:
                    print("\n")
                    data = parse_output_log(run_id)
                    print(format_training_curve(data))
        else:
            print("No latest run found")
    elif args.recent:
        runs = list_runs(limit=args.recent)
        print(format_run_list(runs))
        print("\n")
        for run in runs:
            if run["has_summary"]:
                summary = load_summary(run["wandb_id"])
                if summary:
                    print(f"\n--- {run['wandb_id']} ({run['date']}) ---\n")
                    print(format_run_summary(summary))
    else:
        # Default: list recent runs
        limit = args.list if args.list > 0 else 15
        runs = list_runs(limit=limit)
        print(format_run_list(runs))


if __name__ == "__main__":
    main()
