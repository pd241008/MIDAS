"""
Shared utility for saving experiment results to timestamped JSON files.

Every script that runs attack sweeps must call save_results() before exiting.
Saved files go to shadow/results/<script_name>_<timestamp>.json
"""
import json
import os
import subprocess
from datetime import datetime, timezone


def get_git_commit():
    """Get current git short hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_results(script_name, config, results, extra=None, result_dir=None):
    """
    Save experiment results to a JSON file.

    Args:
        script_name: Name of the script (e.g. "check_trained_surrogate.py")
        config: Dict of experiment config (gamma, lambda, N, etc.)
        results: List of result dicts, each with keys like
                 update_rule, T, epsilon, naive_asr, adaptive_asr, n, gap, gap_se
        extra: Optional dict of additional metadata to include
        result_dir: Override directory for saved files (default: shadow/results/)
    """
    if result_dir is None:
        result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_commit = get_git_commit()

    record = {
        "timestamp": timestamp,
        "script": script_name,
        "git_commit": git_commit,
        "config": config,
        "results": results,
    }
    if extra:
        record["extra"] = extra

    filename = f"{script_name.replace('.py', '')}_{timestamp}.json"
    filepath = os.path.join(result_dir, filename)

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n  Results saved to: {filepath}")
    return filepath
