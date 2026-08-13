"""Simulator evaluator interfaces used by circuit examples."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class LineProcessEvaluator:
    """Evaluate vectors through a persistent stdin/stdout worker process.

    The worker must print ``READY`` once, accept one comma-separated vector per
    line, and return one floating-point value per request. A dead worker is
    restarted once. Worker diagnostics go to ``worker.stderr.log`` inside the
    private run directory, leaving stdout reserved for the protocol. The
    parent's conda library directory and ``LD_PRELOAD`` are removed from the
    child environment so a separate simulator environment can load its own
    native libraries.
    """

    def __init__(
        self,
        command: Sequence[str],
        workdir: str | Path,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.command = [str(part) for part in command]
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.environment = dict(environment) if environment is not None else None
        self._process: subprocess.Popen[str] | None = None
        self._stderr = None

    def _spawn(self) -> None:
        self._close_handles()
        env = dict(os.environ)
        if self.environment:
            env.update(self.environment)
        env.pop("LD_PRELOAD", None)
        worker_bin = Path(self.command[0]).resolve().parent
        worker_prefix = worker_bin.parent
        parent_prefix = env.get("CONDA_PREFIX")
        library_paths = [path for path in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if path]
        if parent_prefix:
            parent = str(Path(parent_prefix).resolve())
            library_paths = [
                path for path in library_paths if not str(Path(path).resolve()).startswith(parent)
            ]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
        env["PATH"] = str(worker_bin) + os.pathsep + env.get("PATH", "")
        env["CONDA_PREFIX"] = str(worker_prefix)
        self._stderr = (self.workdir / "worker.stderr.log").open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            env=env,
        )
        assert self._process.stdout is not None
        ready = self._process.stdout.readline().strip()
        if ready != "READY":
            self.close()
            raise RuntimeError(
                f"worker did not become ready (received {ready!r}); "
                f"see {self.workdir / 'worker.stderr.log'}"
            )

    def evaluate(self, vector: np.ndarray) -> float:
        """Return one simulator response, or NaN after two failed attempts."""

        values = np.asarray(vector, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("vector must be a finite one-dimensional array")
        request = ",".join(f"{value:.10g}" for value in values) + "\n"
        for _attempt in range(2):
            if self._process is None or self._process.poll() is not None:
                self._spawn()
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            try:
                self._process.stdin.write(request)
                self._process.stdin.flush()
                response = self._process.stdout.readline().strip()
                if response:
                    return float(response)
            except (BrokenPipeError, OSError, ValueError):
                pass
            self._terminate_process()
        return float("nan")

    def _terminate_process(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=15)
        self._close_handles()

    def _close_handles(self) -> None:
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def close(self) -> None:
        """Ask the worker to exit, then release process and log handles."""

        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write("EXIT\n")
                process.stdin.flush()
                process.wait(timeout=15)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=15)
        self._close_handles()

    def __enter__(self) -> "LineProcessEvaluator":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
