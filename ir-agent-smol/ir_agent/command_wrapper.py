import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any


@dataclass
class WrappedCommandResult:
    role: str
    cmd: str
    cwd: str
    returncode: int
    status: str
    runtime_sec: float
    stdout: str
    stderr: str
    parsed_json: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "cmd": self.cmd,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "status": self.status,
            "runtime_sec": self.runtime_sec,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "parsed_json": self.parsed_json,
        }


def safe_format_template(template: str, payload: dict[str, Any]) -> str:
    fields = [name for _, name, _, _ in Formatter().parse(template) if name]
    values = dict(payload)
    for field in fields:
        raw = str(values.get(field, ""))
        values[field] = shlex.quote(raw)
    return template.format(**values)


def _try_parse_json_from_stdout(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        return None
    return None


def _build_ssh_command(
    cmd: str,
    cwd: str,
    host: str,
    ssh_options: str = "",
) -> str:
    """Wrap a command to execute on a remote host via SSH.

    The remote command first cd's to the working directory (shared filesystem),
    then executes the original command.
    """
    remote_cmd = f"cd {shlex.quote(cwd)} && {cmd}"
    ssh_parts = ["ssh"]
    if ssh_options:
        ssh_parts.append(ssh_options)
    ssh_parts.append(shlex.quote(host))
    ssh_parts.append(shlex.quote(remote_cmd))
    return " ".join(ssh_parts)


def run_wrapped_command(
    *,
    role: str,
    template: str,
    payload: dict[str, Any],
    cwd: str,
    remote_host: str = "",
    remote_ssh_options: str = "",
) -> dict[str, Any]:
    cmd = safe_format_template(template, payload)

    # If remote execution is configured, wrap via SSH
    if remote_host:
        actual_cmd = _build_ssh_command(cmd, cwd, remote_host, remote_ssh_options)
    else:
        actual_cmd = cmd

    start = time.perf_counter()
    proc = subprocess.run(
        actual_cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd if not remote_host else None,  # cwd is handled by SSH cd
    )
    runtime_sec = round(time.perf_counter() - start, 4)

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    status = "ok" if proc.returncode == 0 else "error"
    parsed = _try_parse_json_from_stdout(stdout)

    result = WrappedCommandResult(
        role=role,
        cmd=cmd,  # log the original command, not the SSH wrapper
        cwd=str(Path(cwd).resolve()),
        returncode=proc.returncode,
        status=status,
        runtime_sec=runtime_sec,
        stdout=stdout,
        stderr=stderr,
        parsed_json=parsed,
    )
    return result.to_dict()
