import json
import base64
import mimetypes
import fnmatch
from pathlib import Path

import requests
from smolagents import tool

from .command_wrapper import run_wrapped_command
from .memory_log import IRTaskMemoryLog
from .settings import settings


memory_log = IRTaskMemoryLog(settings.memory_log_path)

EDITABLE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".sh"}

CURATED_PROJECT_FILES = {
    "fusiontest.py": "Main fusion inference and metric evaluation script.",
    "fusion/models.py": "Fusion model definitions and architectural components.",
    "fusion/losses.py": "Loss functions used by fusion training and evaluation.",
    "fusion/vae_utils.py": "VAE encode/decode helpers used in fusion pipeline.",
    "scripts/train_fusion.py": "Primary training entrypoint for the fusion model.",
    "fusion/trainer.py": "Training loop logic, optimization behavior, freezing strategy, and stage-specific scheduling.",
}


def _run_with_wrapper(role: str, template: str, image_path: str, prompt: str = "") -> dict:
    return run_wrapped_command(
        role=role,
        template=template,
        payload={
            "image_path": image_path,
            "prompt": prompt,
            "infrared_project_root": settings.infrared_project_root,
        },
        cwd=settings.tool_workdir,
        remote_host=settings.remote_gpu_host if settings.remote_gpu_enabled else "",
        remote_ssh_options=settings.remote_ssh_options if settings.remote_gpu_enabled else "",
    )


def _file_to_data_url(path: str) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _safe_project_path(relative_path: str) -> Path:
    project_root = Path(settings.infrared_project_root).resolve()
    target = (project_root / relative_path).resolve()
    if not str(target).startswith(str(project_root)):
        raise ValueError(f"path escapes project root: {relative_path}")
    return target


def _extract_message_text(data: dict) -> str:
    try:
        content = data["choices"][0]["message"].get("content", "")
    except Exception:
        return json.dumps(data, ensure_ascii=False)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join([p for p in parts if p]).strip()
    return str(content)


def _chat_completion(messages: list[dict], *, model_id: str | None = None, timeout_sec: int | None = None) -> dict:
    endpoint = settings.api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id or settings.vision_model_id,
        "messages": messages,
        "stream": False,
    }

    if settings.vision_enable_thinking:
        payload["extra_body"] = {
            "enable_thinking": True,
            "thinking_budget": settings.vision_thinking_budget,
        }

    resp = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=timeout_sec or settings.vision_timeout_sec,
    )
    resp.raise_for_status()
    return resp.json()


@tool
def get_project_code_context(topic: str = "fusion") -> str:
    """Return curated project file map for code-aware reasoning.

    Args:
        topic: high-level topic to focus on, such as fusion, training, losses, or inference.

    Returns:
        JSON with recommended files and their roles inside the infrared project.
    """
    topic_lower = topic.lower()
    recommended = []
    for path, desc in CURATED_PROJECT_FILES.items():
        score = 0
        if topic_lower in path.lower() or topic_lower in desc.lower():
            score += 2
        if any(keyword in topic_lower for keyword in ["train", "trainer", "finetune", "fine-tune", "微调", "训练"]):
            if path in {"fusion/trainer.py", "scripts/train_fusion.py", "fusion/losses.py"}:
                score += 4
        if any(keyword in topic_lower for keyword in ["model", "architecture", "结构", "模块"]):
            if path in {"fusion/models.py", "fusion/trainer.py"}:
                score += 3
        if topic_lower in {"fusion", "training", "losses", "inference"}:
            score += 1
        recommended.append({"path": path, "description": desc, "score": score})

    recommended.sort(key=lambda item: item["score"], reverse=True)
    payload = {
        "project_root": settings.infrared_project_root,
        "topic": topic,
        "files": recommended,
    }
    memory_log.append({"tool": "get_project_code_context", "topic": topic, "notes": "code_context"})
    return json.dumps(payload, ensure_ascii=False)


@tool
def read_project_file(relative_path: str, start_line: int = 1, end_line: int = 220) -> str:
    """Read a code file from the infrared project root for code-aware analysis.

    Args:
        relative_path: file path relative to INFRARED_PROJECT_ROOT.
        start_line: 1-based starting line number.
        end_line: 1-based ending line number.

    Returns:
        JSON with resolved path and requested file content excerpt.
    """
    try:
        target = _safe_project_path(relative_path)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

    if not target.exists() or not target.is_file():
        return json.dumps({"status": "error", "error": f"file not found: {relative_path}"}, ensure_ascii=False)

    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line

    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    excerpt = lines[start_line - 1 : end_line]
    payload = {
        "status": "ok",
        "relative_path": relative_path,
        "resolved_path": str(target),
        "start_line": start_line,
        "end_line": min(end_line, len(lines)),
        "content": "\n".join(excerpt),
    }
    memory_log.append(
        {
            "tool": "read_project_file",
            "relative_path": relative_path,
            "start_line": start_line,
            "end_line": end_line,
            "notes": "code_read",
        }
    )
    return json.dumps(payload, ensure_ascii=False)


@tool
def list_project_files(pattern: str = "**/*.py", limit: int = 300) -> str:
    """List files in infrared project root by glob pattern.

    Args:
        pattern: glob pattern relative to INFRARED_PROJECT_ROOT.
        limit: max number of paths returned.

    Returns:
        JSON with matched relative paths.
    """
    root = Path(settings.infrared_project_root).resolve()
    matches: list[str] = []
    max_items = max(1, min(limit, 2000))

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if fnmatch.fnmatch(rel, pattern):
            matches.append(rel)
            if len(matches) >= max_items:
                break

    memory_log.append({"tool": "list_project_files", "pattern": pattern, "count": len(matches), "notes": "code_list"})
    return json.dumps({"project_root": str(root), "pattern": pattern, "items": matches}, ensure_ascii=False)


@tool
def apply_project_edit(
    relative_path: str,
    old_text: str,
    new_text: str,
    expected_count: int = 1,
    dry_run: bool = False,
) -> str:
    """Apply controlled text replacement in a project file.

    Args:
        relative_path: target file path relative to INFRARED_PROJECT_ROOT.
        old_text: exact text to replace.
        new_text: replacement text.
        expected_count: required replacement count, defaults to 1.
        dry_run: if true, only report how many matches found.

    Returns:
        JSON result with replacement status.
    """
    try:
        target = _safe_project_path(relative_path)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

    if not target.exists() or not target.is_file():
        return json.dumps({"status": "error", "error": f"file not found: {relative_path}"}, ensure_ascii=False)

    if target.suffix.lower() not in EDITABLE_SUFFIXES:
        return json.dumps(
            {"status": "error", "error": f"unsupported suffix: {target.suffix}", "allowed": sorted(EDITABLE_SUFFIXES)},
            ensure_ascii=False,
        )

    content = target.read_text(encoding="utf-8", errors="ignore")
    found = content.count(old_text)
    if found != expected_count:
        return json.dumps(
            {
                "status": "error",
                "error": "match_count_mismatch",
                "expected_count": expected_count,
                "found_count": found,
                "relative_path": relative_path,
            },
            ensure_ascii=False,
        )

    if dry_run:
        return json.dumps(
            {
                "status": "ok",
                "dry_run": True,
                "relative_path": relative_path,
                "found_count": found,
            },
            ensure_ascii=False,
        )

    updated = content.replace(old_text, new_text, expected_count)
    target.write_text(updated, encoding="utf-8")

    payload = {
        "tool": "apply_project_edit",
        "relative_path": relative_path,
        "expected_count": expected_count,
        "found_count": found,
        "dry_run": False,
        "notes": "code_edit",
    }
    memory_log.append(payload)
    return json.dumps({"status": "ok", **payload}, ensure_ascii=False)


@tool
def analyze_image_with_vlm(image_path: str, prompt: str = "") -> str:
    """Analyze image with multimodal model via OpenAI-compatible API.

    Args:
        image_path: absolute path to image.
        prompt: task prompt from user.

    Returns:
        JSON with multimodal diagnosis text and routing hints.
    """
    img = Path(image_path)
    if not img.exists():
        return json.dumps({"error": f"image not found: {image_path}"}, ensure_ascii=False)

    try:
        image_data_url = _file_to_data_url(image_path)
    except Exception as e:
        return json.dumps({"error": f"failed to read image: {e}"}, ensure_ascii=False)

    content_text = (
        "You are an infrared diagnosis assistant. "
        "Given the image and user prompt, output concise diagnosis with: "
        "1) likely degradations, 2) suggested route (fast/slow), 3) suggested first tool (single/mixed), "
        f"4) key parameter hints. User prompt: {prompt}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": content_text},
            ],
        }
    ]

    try:
        data = _chat_completion(messages, model_id=settings.vision_model_id, timeout_sec=settings.vision_timeout_sec)
    except Exception as e:
        error_payload = {
            "tool": "analyze_image_with_vlm",
            "image_path": image_path,
            "prompt": prompt,
            "error": str(e),
            "notes": "vlm_diagnosis_failed",
        }
        memory_log.append(error_payload)
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    text = _extract_message_text(data)

    result = {
        "status": "ok",
        "model": settings.vision_model_id,
        "diagnosis": text,
    }
    memory_log.append(
        {
            "tool": "analyze_image_with_vlm",
            "image_path": image_path,
            "prompt": prompt,
            "model": settings.vision_model_id,
            "notes": "vlm_diagnosis",
        }
    )
    return json.dumps(result, ensure_ascii=False)


@tool
def analyze_fusion_triplet_with_vlm(ir_image_path: str, vis_image_path: str, fused_image_path: str, prompt: str = "") -> str:
    """Analyze IR/Visible/Fused triplet and return actionable fusion diagnosis.

    Args:
        ir_image_path: absolute path to infrared image.
        vis_image_path: absolute path to visible image.
        fused_image_path: absolute path to fused output image.
        prompt: user task prompt.

    Returns:
        JSON diagnosis with artifacts, likely causes, and concrete fixes.
    """
    for p in [ir_image_path, vis_image_path, fused_image_path]:
        if not Path(p).exists():
            return json.dumps({"status": "error", "error": f"image not found: {p}"}, ensure_ascii=False)

    try:
        ir_url = _file_to_data_url(ir_image_path)
        vis_url = _file_to_data_url(vis_image_path)
        fused_url = _file_to_data_url(fused_image_path)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"failed to read triplet image: {e}"}, ensure_ascii=False)

    text_prompt = (
        "You are an expert in infrared-visible image fusion quality analysis. "
        "Input includes infrared source, visible source, and fused result. "
        "Please output in concise JSON style fields: degradations, root_causes, route, first_tool, "
        "metric_hints, training_hints, inference_hints. "
        f"User task: {prompt}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image A is infrared source."},
                {"type": "image_url", "image_url": {"url": ir_url}},
                {"type": "text", "text": "Image B is visible source."},
                {"type": "image_url", "image_url": {"url": vis_url}},
                {"type": "text", "text": "Image C is fused result."},
                {"type": "image_url", "image_url": {"url": fused_url}},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    try:
        data = _chat_completion(messages, model_id=settings.vision_model_id, timeout_sec=settings.vision_timeout_sec)
        diagnosis = _extract_message_text(data)
    except Exception as e:
        memory_log.append(
            {
                "tool": "analyze_fusion_triplet_with_vlm",
                "ir_image_path": ir_image_path,
                "vis_image_path": vis_image_path,
                "fused_image_path": fused_image_path,
                "prompt": prompt,
                "error": str(e),
                "notes": "triplet_diagnosis_failed",
            }
        )
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

    payload = {
        "status": "ok",
        "model": settings.vision_model_id,
        "diagnosis": diagnosis,
    }
    memory_log.append(
        {
            "tool": "analyze_fusion_triplet_with_vlm",
            "ir_image_path": ir_image_path,
            "vis_image_path": vis_image_path,
            "fused_image_path": fused_image_path,
            "prompt": prompt,
            "model": settings.vision_model_id,
            "notes": "triplet_diagnosis",
        }
    )
    return json.dumps(payload, ensure_ascii=False)


@tool
def propose_fusion_improvements(diagnosis: str, code_context: str = "", prompt: str = "") -> str:
    """Generate actionable fusion improvement plan from diagnosis and code context.

    Args:
        diagnosis: text or JSON diagnosis from VLM tools.
        code_context: optional snippets from read_project_file/get_project_code_context.
        prompt: original user request.

    Returns:
        JSON with prioritized improvements and validation checklist.
    """
    planner_prompt = (
        "You are a senior infrared-visible fusion engineer. "
        "Given diagnosis and code context, output a compact plan with fields: "
        "priority_fixes, why, expected_gain, risk, validation_steps. "
        "Prefer practical code-level changes. "
        f"User task: {prompt}\n"
        f"Diagnosis:\n{diagnosis}\n"
        f"Code context:\n{code_context}"
    )

    try:
        data = _chat_completion(
            messages=[{"role": "user", "content": planner_prompt}],
            model_id=settings.model_id,
            timeout_sec=settings.vision_timeout_sec,
        )
        plan_text = _extract_message_text(data)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

    payload = {"status": "ok", "model": settings.model_id, "plan": plan_text}
    memory_log.append(
        {
            "tool": "propose_fusion_improvements",
            "prompt": prompt,
            "notes": "improvement_plan",
        }
    )
    return json.dumps(payload, ensure_ascii=False)


@tool
def classify_prompt_complexity(prompt: str) -> str:
    """Classify prompt complexity for routing.

    Args:
        prompt: user text prompt.

    Returns:
        JSON string with route field: fast or slow.
    """
    direct_keywords = [
        "denoise",
        "deblur",
        "去噪",
        "去模糊",
        "增强对比",
        "低照增强",
    ]
    route = "fast" if any(k in prompt.lower() for k in direct_keywords) else "slow"
    memory_log.append(
        {
            "tool": "classify_prompt_complexity",
            "prompt": prompt,
            "route": route,
            "notes": "route_decision",
        }
    )
    return json.dumps({"route": route}, ensure_ascii=False)


@tool
def run_single_restoration_tool(image_path: str, prompt: str = "") -> str:
    """Run single-distortion infrared restoration tool.

    Args:
        image_path: absolute path of input image.
        prompt: user request text used as additional execution context.

    Returns:
        JSON execution result.
    """
    if not Path(image_path).exists():
        return json.dumps({"error": f"image not found: {image_path}"}, ensure_ascii=False)
    result = _run_with_wrapper("single", settings.single_tool_cmd, image_path, prompt)
    memory_log.append(
        {
            "tool": "run_single_restoration_tool",
            "image_path": image_path,
            "prompt": prompt,
            "returncode": result.get("returncode", -1),
            "notes": "single_distortion_restoration",
        }
    )
    return json.dumps(result, ensure_ascii=False)


@tool
def run_mixed_restoration_tool(image_path: str, prompt: str = "") -> str:
    """Run mixed-distortion infrared restoration tool.

    Args:
        image_path: absolute path of input image.
        prompt: user request text used as additional execution context.

    Returns:
        JSON execution result.
    """
    if not Path(image_path).exists():
        return json.dumps({"error": f"image not found: {image_path}"}, ensure_ascii=False)
    result = _run_with_wrapper("mixed", settings.mixed_tool_cmd, image_path, prompt)
    memory_log.append(
        {
            "tool": "run_mixed_restoration_tool",
            "image_path": image_path,
            "prompt": prompt,
            "returncode": result.get("returncode", -1),
            "notes": "mixed_distortion_restoration",
        }
    )
    return json.dumps(result, ensure_ascii=False)


@tool
def run_feedback_check(image_path: str, history: str = "") -> str:
    """Evaluate current restoration output and tell if should continue.

    Args:
        image_path: absolute path of current image.
        history: short text of previous tool calls.

    Returns:
        JSON with should_continue boolean and raw evaluation result.
    """
    if not Path(image_path).exists():
        return json.dumps({"error": f"image not found: {image_path}"}, ensure_ascii=False)

    result = _run_with_wrapper("eval", settings.eval_tool_cmd, image_path, history)
    text = (result.get("stdout") or "").lower()

    negative_flags = ["bad", "fail", "artifact", "not clean", "need continue"]
    should_continue = any(flag in text for flag in negative_flags)

    payload = {
        "should_continue": should_continue,
        "history": history,
        "evaluation": result,
    }
    memory_log.append(
        {
            "tool": "run_feedback_check",
            "image_path": image_path,
            "history": history,
            "should_continue": should_continue,
            "returncode": result.get("returncode", -1),
            "notes": "quality_gate",
        }
    )
    return json.dumps(payload, ensure_ascii=False)


@tool
def query_task_memory(query: str, limit: int = 5) -> str:
    """Query historical infrared task memory for workflow reuse.

    Args:
        query: keywords to match past prompts/tools/results.
        limit: maximum records to return.

    Returns:
        JSON list of matched memory records.
    """
    rows = memory_log.search(query=query, limit=limit)
    return json.dumps({"query": query, "items": rows}, ensure_ascii=False)


@tool
def summarize_task_memory(limit: int = 200) -> str:
    """Summarize historical infrared task memory and suggest workflow tuning.

    Args:
        limit: number of latest records to analyze.

    Returns:
        JSON summary with per-tool success stats and quick suggestions.
    """
    rows = memory_log.recent(limit=limit)
    if not rows:
        return json.dumps(
            {
                "total": 0,
                "suggestions": ["No memory yet. Run a few tasks first."],
            },
            ensure_ascii=False,
        )

    stats: dict[str, dict[str, float]] = {}
    for row in rows:
        tool_name = str(row.get("tool", "unknown"))
        item = stats.setdefault(
            tool_name,
            {
                "count": 0,
                "ok": 0,
                "need_continue": 0,
            },
        )
        item["count"] += 1
        rc = row.get("returncode")
        if isinstance(rc, int) and rc == 0:
            item["ok"] += 1
        if row.get("should_continue") is True:
            item["need_continue"] += 1

    suggestions: list[str] = []
    for tool_name, s in stats.items():
        count = max(1, int(s["count"]))
        ok_rate = s["ok"] / count
        continue_rate = s["need_continue"] / count
        if ok_rate < 0.8:
            suggestions.append(
                f"{tool_name}: low success rate ({ok_rate:.2f}), check command template/paths/weights."
            )
        if tool_name == "run_feedback_check" and continue_rate > 0.5:
            suggestions.append(
                "High continue rate in quality gate, consider stronger mixed restoration or tuned checkpoints."
            )

    if not suggestions:
        suggestions.append("Current workflow is stable. Next step: tune model checkpoints for quality lift.")

    return json.dumps(
        {
            "total": len(rows),
            "stats": stats,
            "suggestions": suggestions,
        },
        ensure_ascii=False,
    )
