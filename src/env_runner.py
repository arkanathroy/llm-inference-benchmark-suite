"""Thin subprocess bridge for invoking a script inside one of the isolated
per-technique virtual environments (envs/<name>/venv).

Streams the child process's stdout/stderr live, line by line, as it
happens — rather than buffering with subprocess.run(capture_output=True)
and only showing output once the whole process exits. That buffering
was the root cause of long silent gaps in Colab during benchmark runs:
scripts like benchmark_runner.py were producing progress output the
whole time, but none of it reached the notebook cell until the process
was already finished. All streamed lines are also collected so the
full output is still available/re-printable on failure.
"""

from __future__ import annotations

import subprocess
import sys
import threading
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


def _stream_pipe(pipe, sink, collected):
    for line in iter(pipe.readline, ""):
        sink.write(line)
        sink.flush()
        collected.append(line)
    pipe.close()


def _run(cmd, check):
    print(f"[env_runner] $ {' '.join(cmd)}", file=sys.stderr)
    import os
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, text=True, env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=1,
    )

    stdout_lines, stderr_lines = [], []
    t_out = threading.Thread(target=_stream_pipe, args=(proc.stdout, sys.stdout, stdout_lines))
    t_err = threading.Thread(target=_stream_pipe, args=(proc.stderr, sys.stderr, stderr_lines))
    t_out.start()
    t_err.start()

    returncode = proc.wait()
    t_out.join()
    t_err.join()

    result = subprocess.CompletedProcess(
        cmd, returncode, stdout="".join(stdout_lines), stderr="".join(stderr_lines),
    )

    if check and returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {returncode}: {' '.join(cmd)}\n"
            f"Full stdout/stderr was streamed live above as the process ran."
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
