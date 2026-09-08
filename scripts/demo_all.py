import subprocess
import sys

COMMANDS = [
    [sys.executable, "scripts/demo_preprocessing.py"],
    [sys.executable, "scripts/demo_sentiment.py", "--model", "ef_lstm", "--epochs", "1", "--samples", "96"],
    [sys.executable, "scripts/demo_sentiment.py", "--model", "lf_lstm", "--epochs", "1", "--samples", "96"],
    [sys.executable, "scripts/demo_sentiment.py", "--model", "tfn", "--epochs", "1", "--samples", "96"],
    [sys.executable, "scripts/demo_sentiment.py", "--model", "clip_prompt", "--epochs", "1", "--samples", "96"],
    [sys.executable, "scripts/demo_sentiment.py", "--model", "opt_finetune", "--epochs", "1", "--samples", "96"],
    [sys.executable, "scripts/demo_opt_pretrain.py"],
    [sys.executable, "scripts/demo_continual.py"],
    [sys.executable, "scripts/demo_active.py"],
]

for cmd in COMMANDS:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)
