"""Command-line entry point for the dialogue agents.

Examples
--------
Run a single dialogue round on the existing successful episode::

    python -m dialogue_sensor.cli \\
        --episode_dir /home/jc166/dialogue-as-a-sensor/data-generation/my_data/episode_00000 \\
        --trigger_reason semantic_ambiguity

Run on the new stuck-state generator output, with multi-turn dialogue and
Agent 3 grounding (default)::

    python -m dialogue_sensor.cli \\
        --episode_dir /home/jc166/dialogue-as-a-sensor/data-generation/my_stuck_data/episode_00003 \\
        --image_filename image_phase_grasp_descend.png \\
        --max_turns 2 \\
        --output_json runs/episode_00003.json

Run only Agent 2 + the simulated user (skip Agent 3)::

    python -m dialogue_sensor.cli --episode_dir <dir> --no_grounding

Run a smoke test that does NOT call any LLMs (renders prompts only)::

    python -m dialogue_sensor.cli --episode_dir <dir> --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .agents.simulated_user import SIMULATED_USER_SYSTEM, _format_user_prompt as _user_prompt_for_user
from .agents.spatial_grounding import (
    SPATIAL_GROUNDING_SYSTEM,
    _format_user_prompt as _user_prompt_for_grounding,
)
from .agents.visual_inquisitor import (
    VISUAL_INQUISITOR_SYSTEM,
    _format_user_prompt as _user_prompt_for_inquisitor,
)
from .config import Config, load_config
from .episode_loader import load_episode_state
from .graph import build_graph
from .llm_clients import build_client
from .overlays import encode_png, render_overlay


def _format_pose(pose: List[float]) -> str:
    return "[" + ", ".join(f"{float(v):+.3f}" for v in pose) + "]"


def _print_dialogue(final_state: Dict[str, Any]) -> None:
    md = final_state.get("metadata") or {}
    overlay_used = md.get("_overlay_used", False)
    overlay_default_cam = md.get("_overlay_used_default_camera", False)
    print("=" * 72)
    print(f"Task            : {final_state.get('task_description')}")
    print(f"Trigger reason  : {final_state.get('trigger_reason')}")
    print(f"Image           : {final_state.get('image_path')}")
    overlay_label = "yes" if overlay_used else "no"
    if overlay_used and overlay_default_cam:
        overlay_label += " (default K/T_wc)"
    print(f"Overlay         : {overlay_label}")
    print(f"Turns completed : {final_state.get('turn_count')} / {final_state.get('max_turns')}")
    print("-" * 72)
    for t in final_state.get("dialogue_history", []) or []:
        speaker = "ROBOT (Agent 2)" if t["role"] == "robot" else "HUMAN (sim)"
        print(f"[{speaker}] {t['content']}")
    print("=" * 72)
    print(f"Final clarified instruction:\n  {final_state.get('clarified_instruction')}")
    print("=" * 72)


def _print_grounding(final_state: Dict[str, Any]) -> None:
    if "corrections" not in final_state:
        return  # grounding skipped
    md = final_state.get("metadata") or {}
    original = md.get("waypoints") or {}
    corrected = final_state.get("corrected_waypoints") or {}
    corrections = final_state.get("corrections") or {}
    reasoning = final_state.get("grounding_reasoning")
    osc_delta = final_state.get("osc_delta")

    print()
    print("AGENT 3 (Spatial-to-OSC Grounding)")
    print("=" * 72)
    if reasoning:
        print(f"Reasoning       : {reasoning}")
    if not corrections:
        print("Corrections     : (none — original waypoints already correct)")
    else:
        print("Corrections (per waypoint, deltas in m / rad / dimensionless gripper):")
        print(f"  {json.dumps(corrections, indent=2)}")
    if osc_delta is not None:
        formatted = ", ".join(f"{v:+.4f}" for v in osc_delta)
        print(f"A2 OSC Δ vector : [{formatted}]  (m, m, m, rad, rad, rad, gripper)")
    if corrections and original and corrected:
        print("Waypoints (original -> corrected):")
        for key in ("A1_pregrasp", "A2_grasp", "A3_release"):
            if key not in original:
                continue
            orig = original[key]
            new = corrected.get(key, orig)
            short = key.split("_", 1)[0]
            if orig == new:
                print(f"  {short} ({key}): {_format_pose(orig)}  (unchanged)")
            else:
                print(f"  {short} ({key}):")
                print(f"     orig: {_format_pose(orig)}")
                print(f"     new : {_format_pose(new)}")
    print("=" * 72)


def _save_corrected_overlay(
    final_state: Dict[str, Any],
    image_path: str,
    save_to: Path,
) -> None:
    """Render the *corrected* waypoints onto the source image and save."""
    corrected = final_state.get("corrected_waypoints") or {}
    if not corrected:
        return
    md = final_state.get("metadata") or {}
    K = md.get("camera_intrinsics")
    T_wc = md.get("camera_extrinsics_wc")
    import numpy as np  # local import to avoid mandatory cost when unused
    K_arr = np.asarray(K, dtype=np.float64) if K is not None else None
    T_arr = np.asarray(T_wc, dtype=np.float64) if T_wc is not None else None
    img_bytes = Path(image_path).read_bytes()
    annotated = render_overlay(img_bytes, corrected, K=K_arr, T_wc=T_arr)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    save_to.write_bytes(encode_png(annotated))
    print(f"Saved corrected-waypoint overlay to {save_to}")


def _dry_run(state) -> None:
    """Render the prompts each agent would send, without hitting any API."""
    print("=" * 72)
    print("DRY RUN — no API calls will be made.")
    print("=" * 72)
    print(">>> Visual Inquisitor (Agent 2) — system:")
    print(VISUAL_INQUISITOR_SYSTEM)
    print()
    print(">>> Visual Inquisitor (Agent 2) — user prompt:")
    print(_user_prompt_for_inquisitor(state))
    print()
    print(">>> Simulated User — system:")
    print(SIMULATED_USER_SYSTEM)
    print()
    print(">>> Simulated User — user prompt (assuming Agent 2 already asked a question):")
    placeholder_state = dict(state)
    placeholder_state["clarifying_question"] = "<Agent 2 question goes here>"
    placeholder_state["dialogue_history"] = list(placeholder_state.get("dialogue_history", []) or []) + [
        {"role": "robot", "content": "<Agent 2 question goes here>"}
    ]
    print(_user_prompt_for_user(placeholder_state))
    print()
    print(">>> Spatial Grounding (Agent 3) — system:")
    print(SPATIAL_GROUNDING_SYSTEM)
    print()
    print(
        ">>> Spatial Grounding (Agent 3) — user prompt (assuming the dialogue "
        "concluded with a clarified instruction):"
    )
    placeholder_state["dialogue_history"] = list(placeholder_state["dialogue_history"]) + [
        {"role": "human", "content": "<simulated user's clarified instruction goes here>"}
    ]
    placeholder_state["clarified_instruction"] = "<simulated user's clarified instruction goes here>"
    print(_user_prompt_for_grounding(placeholder_state))
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dialogue-sensor",
        description="Run Agent 2 (Visual Inquisitor), the simulated HITL user, "
                    "and Agent 3 (Spatial-to-OSC Grounding) on a single "
                    "robosuite episode.",
    )
    parser.add_argument(
        "--episode_dir", required=True,
        help="Directory holding image_*.png + metadata.json from the data-generation pipeline.",
    )
    parser.add_argument(
        "--image_filename", default=None,
        help="Override which image inside the episode dir to use (e.g. image_phase_grasp_descend.png).",
    )
    parser.add_argument(
        "--trigger_reason",
        choices=["semantic_ambiguity", "kinematic_collision", "unknown"],
        default=None,
        help="Override the trigger reason. Defaults to metadata.json -> trigger_reason.",
    )
    parser.add_argument(
        "--max_turns", type=int, default=None,
        help="Maximum number of dialogue turns. Defaults to MAX_DIALOGUE_TURNS env var.",
    )
    parser.add_argument(
        "--output_json", default=None,
        help="Optional path to dump the final state (dialogue history + Agent 3 outputs) as JSON.",
    )
    parser.add_argument(
        "--no_overlay", action="store_true",
        help="Disable the planned-trajectory overlay (A1/A2/A3 axis frames). "
             "By default the overlay is rendered onto the agentview image.",
    )
    parser.add_argument(
        "--save_overlay_png", default=None,
        help="Optional path to save the annotated overlay PNG (for inspection). "
             "Defaults to <output_json>.overlay.png if --output_json is given.",
    )
    parser.add_argument(
        "--no_grounding", action="store_true",
        help="Skip Agent 3 (Spatial-to-OSC Grounding). The graph stops after "
             "the dialogue loop. Useful for ablations / debugging Agent 2.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Render the prompts without calling any LLM. No API keys required.",
    )
    args = parser.parse_args(argv)

    cfg: Config = load_config()
    max_turns = args.max_turns if args.max_turns is not None else cfg.max_turns
    enable_grounding = cfg.enable_spatial_grounding and not args.no_grounding

    overlay_save_path = args.save_overlay_png
    if overlay_save_path is None and args.output_json:
        overlay_save_path = str(Path(args.output_json).with_suffix(".overlay.png"))

    state = load_episode_state(
        episode_dir=args.episode_dir,
        image_filename=args.image_filename,
        trigger_reason=args.trigger_reason,
        max_turns=max_turns,
        overlay=not args.no_overlay,
        save_overlay_to=overlay_save_path,
    )

    if args.dry_run:
        _dry_run(state)
        return 0

    # Validate only the providers we'll actually invoke.
    if not enable_grounding:
        # Defensive copy that flips the flag for validate(), so we don't ask
        # for an OPENAI key the user hasn't set if they opted out of Agent 3.
        cfg.enable_spatial_grounding = False
    cfg.validate()

    inquisitor = build_client(
        provider=cfg.visual_inquisitor_provider,
        api_key=cfg.openai_api_key if cfg.visual_inquisitor_provider == "openai" else cfg.google_api_key,
        model=cfg.visual_inquisitor_model,
    )
    sim_user = build_client(
        provider=cfg.simulated_user_provider,
        api_key=cfg.openai_api_key if cfg.simulated_user_provider == "openai" else cfg.google_api_key,
        model=cfg.simulated_user_model,
    )
    grounding = None
    if enable_grounding:
        grounding = build_client(
            provider=cfg.spatial_grounding_provider,
            api_key=cfg.openai_api_key if cfg.spatial_grounding_provider == "openai" else cfg.google_api_key,
            model=cfg.spatial_grounding_model,
        )

    graph = build_graph(
        inquisitor_client=inquisitor,
        user_client=sim_user,
        grounding_client=grounding,
        verbose=cfg.verbose,
    )
    final_state = graph.invoke(state)

    _print_dialogue(final_state)
    if enable_grounding:
        _print_grounding(final_state)

    # Side-by-side corrected overlay PNG.
    if args.output_json and final_state.get("corrections"):
        corrected_overlay = Path(args.output_json).with_suffix(".corrected.overlay.png")
        try:
            _save_corrected_overlay(
                final_state,
                image_path=final_state["image_path"],
                save_to=corrected_overlay,
            )
        except Exception as e:
            print(f"[cli] WARN: failed to render corrected-overlay PNG: {e}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Strip the giant base64 image from the dump - keep the path instead.
        dump = {k: v for k, v in final_state.items() if k != "image_b64"}
        out.write_text(json.dumps(dump, indent=2, default=str))
        print(f"Saved dialogue+grounding trace to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
