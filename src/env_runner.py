"""Thin subprocess bridge for invoking a script inside one of the isolated
per-technique virtual environments (envs/<name>/venv).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def venv_python(env_name: str) -> Path:
    py = REPO_ROOT / "envs" / env_name / "venv" / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(
            f"No venv found for '{env_name}' at {py}. "
            f"Run `bash envs/{env_name}/setup.sh` first."
        )
    return py


def run_in_env(env_name, script_path, args=None, check=True, capture_output=False):
    py = venv_python(env_name)
    cmd = [str(py), str(script_path)] + (args or [])
    print(f"[env_runner] ({env_name}) $ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=check, capture_output=capture_output, text=True)


def run_module_in_env(env_name, module, args=None, check=True, capture_output=False):
    py = venv_python(env_name)
    cmd = [str(py), "-m", module] + (args or [])
    print(f"[env_runner] ({env_name}) $ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=check, capture_output=capture_output, text=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a script inside an isolated env")
    parser.add_argument("env_name", choices=["fp16", "gptq", "awq", "gguf", "trtllm"])
    parser.add_argument("script_path")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    run_in_env(ns.env_name, ns.script_path, ns.script_args)
