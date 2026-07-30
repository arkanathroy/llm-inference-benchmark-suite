"""Thin subprocess bridge for invoking a script inside one of the isolated
per-technique virtual environments (envs/<name>/venv).

IMPORTANT: subprocess stdout/stderr do not reliably surface in Colab's
notebook cell output the way `!shell` magic does (Colab only captures the
parent kernel's sys.stdout, not raw OS file descriptors inherited by a
child process). To avoid silently losing the real traceback on failure,
this module always captures output internally and re-prints it explicitly
before raising, so errors are visible directly in the cell that called
run_in_env instead of only showing a generic CalledProcessError.
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


def _run(cmd, check):
    print(f"[env_runner] $ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}\n"
            f"See captured stdout/stderr printed above for the real traceback."
        )
    return result


def run_in_env(env_name, script_path, args=None, check=True, capture_output=False):
    py = venv_python(env_name)
    cmd = [str(py), str(script_path)] + (args or [])
    return _run(cmd, check)


def run_module_in_env(env_name, module, args=None, check=True, capture_output=False):
    py = venv_python(env_name)
    cmd = [str(py), "-m", module] + (args or [])
    return _run(cmd, check)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a script inside an isolated env")
    parser.add_argument("env_name", choices=["fp16", "gptq", "awq", "gguf", "trtllm"])
    parser.add_argument("script_path")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    run_in_env(ns.env_name, ns.script_path, ns.script_args)
