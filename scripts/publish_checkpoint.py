
import shutil
import subprocess
from pathlib import Path
import sys


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python scripts/publish_checkpoint.py RUN_DIR"
    )


run_dir = Path(sys.argv[1])

if not run_dir.exists():
    raise FileNotFoundError(
        f"Run directory does not exist: {run_dir}"
    )


required_files = [
    "latest.pt",
    "best.pt",
    "log.jsonl",
    "train.yaml",
]

for filename in required_files:
    path = run_dir / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required checkpoint artifact: {path}"
        )


output_dir = Path("/kaggle/working/m46-checkpoints")
output_dir.mkdir(parents=True, exist_ok=True)


destination = output_dir / run_dir.name

if destination.exists():
    shutil.rmtree(destination)


shutil.copytree(run_dir, destination)

print(f"Published local copy: {destination}")


subprocess.run(
    [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(output_dir),
        "--dir-mode",
        "zip",
        "-m",
        f"M4.6 checkpoint update: {run_dir.name}",
    ],
    check=True,
)


print("Checkpoint successfully published to Kaggle Dataset.")