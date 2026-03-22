import argparse
from pathlib import Path

import yaml
from smolagents import CodeAgent, OpenAIServerModel

from .settings import settings
from .tools import (
    analyze_fusion_triplet_with_vlm,
    analyze_image_with_vlm,
    apply_project_edit,
    classify_prompt_complexity,
    get_project_code_context,
    list_project_files,
    propose_fusion_improvements,
    read_project_file,
    query_task_memory,
    summarize_task_memory,
    run_feedback_check,
    run_mixed_restoration_tool,
    run_single_restoration_tool,
)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(config: dict) -> CodeAgent:
    model = OpenAIServerModel(
        model_id=settings.model_id,
        api_base=settings.api_base,
        api_key=settings.api_key,
    )

    prompt = config["agent"]["system_prompt"]
    return CodeAgent(
        tools=[
            analyze_fusion_triplet_with_vlm,
            analyze_image_with_vlm,
            propose_fusion_improvements,
            list_project_files,
            apply_project_edit,
            classify_prompt_complexity,
            get_project_code_context,
            read_project_file,
            query_task_memory,
            summarize_task_memory,
            run_single_restoration_tool,
            run_mixed_restoration_tool,
            run_feedback_check,
        ],
        model=model,
        max_steps=int(config["agent"].get("max_steps", 4)),
        additional_authorized_imports=["json"],
        instructions=prompt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Infrared workflow agent")
    parser.add_argument("--image", help="Absolute path to input image (single-image mode)")
    parser.add_argument("--ir-image", help="Absolute path to infrared source image")
    parser.add_argument("--vis-image", help="Absolute path to visible source image")
    parser.add_argument("--fused-image", help="Absolute path to fused image")
    parser.add_argument("--prompt", required=True, help="User request")
    parser.add_argument(
        "--config",
        default="config/agent.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    triplet_mode = bool(args.ir_image and args.vis_image and args.fused_image)

    image: Path | None = None
    ir_image: Path | None = None
    vis_image: Path | None = None
    fused_image: Path | None = None

    if triplet_mode:
        ir_image = Path(args.ir_image)
        vis_image = Path(args.vis_image)
        fused_image = Path(args.fused_image)
        for p in [ir_image, vis_image, fused_image]:
            if not p.exists():
                raise FileNotFoundError(f"Input image not found: {p}")
    else:
        if not args.image:
            raise ValueError("Provide --image for single-image mode, or provide all of --ir-image --vis-image --fused-image")
        image = Path(args.image)
        if not image.exists():
            raise FileNotFoundError(f"Input image not found: {image}")

    if not settings.api_key:
        raise RuntimeError("Missing API_KEY in environment")

    config = load_config(Path(args.config))
    agent = build_agent(config)

    if triplet_mode:
        assert ir_image and vis_image and fused_image
        task = (
            "User prompt: "
            + args.prompt
            + "\n"
            + f"Infrared source path: {ir_image.resolve()}\n"
            + f"Visible source path: {vis_image.resolve()}\n"
            + f"Fused image path: {fused_image.resolve()}\n"
            + "Required workflow:\n"
            + "1) Run analyze_fusion_triplet_with_vlm first for fusion-specific diagnosis.\n"
            + "2) Query memory using query_task_memory and summarize_task_memory for reusable patterns.\n"
            + "3) Fetch code context using get_project_code_context and read_project_file for likely affected modules.\n"
            + "4) Generate actionable plan with propose_fusion_improvements.\n"
            + "5) If user requests code update, locate files with list_project_files and apply minimal safe edits with apply_project_edit.\n"
            + "6) Output concise final report with image issues, root causes, code changes, and verification checklist."
        )
    else:
        assert image
        task = (
            "User prompt: "
            + args.prompt
            + "\n"
            + f"Input image path: {image.resolve()}\n"
            + "Required workflow:\n"
            + "1) If the user asks for code-aware advice, first call get_project_code_context, then read_project_file for the most relevant files.\n"
            + "2) Run analyze_image_with_vlm for multimodal diagnosis on input image.\n"
            + "3) Query historical memory with query_task_memory based on prompt keywords.\n"
            + "4) Use summarize_task_memory briefly to check recent failure patterns.\n"
            + "5) Decide fast/slow route with classify_prompt_complexity and VLM diagnosis.\n"
            + "6) Fast route: run_single_restoration_tool once.\n"
            + "7) Slow route: run_mixed_restoration_tool first, then run_single_restoration_tool if needed.\n"
            + "8) Always run run_feedback_check.\n"
            + "9) Output concise final report with selected route, code context used, VLM diagnosis, reused memory, tool calls and next action suggestion."
        )

    result = agent.run(task)
    print(result)


if __name__ == "__main__":
    main()
