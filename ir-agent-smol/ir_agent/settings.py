import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFRARED_ROOT = PROJECT_ROOT.parent / "ddpm_demo" / "Thermal_Diffusion_Project"


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    api_base: str = os.getenv("API_BASE", "http://127.0.0.1:8000/v1")
    api_key: str = os.getenv("API_KEY", "")
    model_id: str = os.getenv("MODEL_NAME", "qwen3.5-9b-local")

    infrared_project_root: str = os.getenv("INFRARED_PROJECT_ROOT", str(DEFAULT_INFRARED_ROOT))
    tool_workdir: str = os.getenv("IR_TOOL_WORKDIR", str(DEFAULT_INFRARED_ROOT))

    single_tool_cmd: str = os.getenv(
        "IR_SINGLE_TOOL_CMD",
        "python inference.py",
    )
    mixed_tool_cmd: str = os.getenv(
        "IR_MIXED_TOOL_CMD",
        "python fusiontest.py",
    )
    eval_tool_cmd: str = os.getenv(
        "IR_EVAL_CMD",
        "python fusiontest.py",
    )

    memory_log_path: str = os.getenv(
        "IR_TASK_MEMORY_PATH",
        str(PROJECT_ROOT / "logs" / "ir_task_memory.jsonl"),
    )

    vision_model_id: str = os.getenv("VISION_MODEL_NAME", os.getenv("MODEL_NAME", "qwen3.5-9b-local"))
    vision_enable_thinking: bool = _as_bool(os.getenv("VISION_ENABLE_THINKING"), default=False)
    vision_thinking_budget: int = int(os.getenv("VISION_THINKING_BUDGET", "8192"))
    vision_timeout_sec: int = int(os.getenv("VISION_TIMEOUT_SEC", "120"))

    # Remote execution settings (for running GPU commands on compute nodes)
    remote_gpu_host: str = os.getenv("REMOTE_GPU_HOST", "")  # e.g. "gpu01" or "user@gpu01"
    remote_gpu_enabled: bool = _as_bool(os.getenv("REMOTE_GPU_ENABLED"), default=False)
    remote_ssh_options: str = os.getenv("REMOTE_SSH_OPTIONS", "-o StrictHostKeyChecking=no -o ConnectTimeout=10")


settings = Settings()
