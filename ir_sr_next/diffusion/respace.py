from __future__ import annotations


def space_timesteps(num_steps: int, respacing: str) -> list[int]:
    """Support formats like 'ddim50' or '50'."""
    if respacing.startswith("ddim"):
        target = int(respacing.replace("ddim", ""))
    else:
        target = int(respacing)

    if target <= 1 or target > num_steps:
        raise ValueError(f"Invalid respacing target: {target} for num_steps={num_steps}")

    stride = (num_steps - 1) / (target - 1)
    return sorted({round(i * stride) for i in range(target)})
