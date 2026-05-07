"""Agent 2 — the Visual Inquisitor (NLG node).

Looks at the paused-simulation image plus the high-level task instruction and
asks the human supervisor exactly one clarifying question per turn. The
answer comes back through the simulated-user node.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..llm_clients import ChatMessage, ImageInput, VisionLLMClient
from ..state import DialogueState, DialogueTurn


VISUAL_INQUISITOR_SYSTEM = """You are the visual inquisitor inside a robotic-manipulation pipeline.

A Panda robot arm is performing a tabletop pick-and-place task in Robosuite.
Its low-level controller has hit an uncertainty trigger and the simulation
is paused. You have been given:
  - the current camera view of the workspace, augmented with the robot's
    PLANNED action waypoints overlaid as 3D coordinate frames, and
  - the high-level natural-language task description.

The overlay you see on the image is the robot's intended action plan:
  - "A1" labels the pre-grasp waypoint (hover above the target).
  - "A2" labels the grasp waypoint (descend + close gripper on the target).
  - "A3" labels the release waypoint (above the target bin).
At each waypoint a 3D coordinate frame is drawn:
  - RED line  = X-axis. The gripper's "pinch" direction; positive X points
    away from the camera.
  - GREEN line = Y-axis. Positive Y goes to the viewer's RIGHT, negative Y
    to the LEFT.
  - BLUE line = Z-axis. Vertical; positive Z points UP.
The 7-DoF coordinates of each waypoint are also given in the prompt as
[x, y, z, roll, pitch, yaw, gripper] in the world frame, with units:
  - x, y, z      : METERS (m). The tabletop workspace spans roughly
                   x ∈ [-0.10, +0.30] m, y ∈ [-0.50, +0.55] m,
                   z ∈ [+0.80, +1.20] m. The cereal box is ~10 cm long;
                   typical correction nudges are 0.02-0.05 m (2-5 cm).
  - roll, pitch, yaw : RADIANS, applied as scipy 'xyz' Euler angles.
                   90° = π/2 ≈ 1.5708 rad; 180° = π ≈ 3.1416 rad. A grasp
                   pointing straight down has roll ≈ ±π and pitch ≈ 0.
  - gripper      : dimensionless in [-1, +1]. -1 = fully OPEN,
                   +1 = fully CLOSED.

Visual geometry rule for a successful grasp:
  - GOOD ALIGNMENT: at A1 and A2, the RED line points perpendicularly INTO
    the broad, flat face of the target object, and the GREEN line runs
    parallel to the target's long edge.
  - BAD ALIGNMENT: the RED line is parallel to the long edge (the gripper
    will smash into the object's narrow side), or the axes are diagonal.
  - The gripper PINCH direction is RED, so a grasp succeeds only when RED is
    perpendicular to the broad face.

Your job is to ask the human supervisor ONE focused, actionable clarifying
question that will let the robot resume.

Rules:
- Output ONLY the question and nothing else.
- Ground every question in something you can actually see in the image —
  named objects (with colour / position) AND, when relevant, the overlaid
  A1/A2/A3 frames or their axis orientations.
- Prefer binary or small-multiple-choice questions. Examples:
  - "Should I keep A1's current orientation or rotate its yaw by π/2 rad
    (≈1.571 rad) so the red axis is perpendicular to the cereal's broad face?"
  - "Should I push the green bottle aside, or shift A2 0.05 m to the left to
    avoid it?"
  - "Is the cereal the orange box A2 is targeting, or the red box behind it?"
- Respect the trigger reason:
    * semantic_ambiguity → confusion about WHAT to grasp / where to place.
      Use the overlay to point at A2 / A3 specifically.
    * kinematic_collision → an obstruction is blocking the planned motion;
      ask about how to reroute, push, or re-orient (often: rotate A1's red
      axis, or nudge A2's xy/z).
    * unknown → ask the most useful disambiguating question for the scene.
- Keep the question under 35 words.
- If a previous human reply already gave enough information, ask a brief
  confirmation rather than a redundant disambiguation."""


def _format_user_prompt(state: DialogueState) -> str:
    """Render the per-turn user prompt seen by the inquisitor model."""
    task = state.get("task_description", "(no task description provided)")
    trigger = state.get("trigger_reason", "unknown")
    metadata: Dict[str, Any] = state.get("metadata", {}) or {}
    scene_objects = metadata.get("scene_objects") or []
    stuck_phase = metadata.get("stuck_phase")
    waypoints = metadata.get("waypoints") or {}
    overlay_used = bool(metadata.get("_overlay_used", False))
    pixel_summary = metadata.get("_overlay_waypoint_pixels") or []

    lines: list[str] = [
        f"Task: {task}",
        f"Trigger reason: {trigger}",
    ]
    if stuck_phase:
        lines.append(f"Trajectory phase where the robot stalled: {stuck_phase}")
    if scene_objects:
        lines.append(
            "Objects expected in the source bin (ground-truth, may be occluded in "
            f"the image): {', '.join(scene_objects)}"
        )

    if overlay_used and waypoints:
        lines.append(
            "The image has the robot's planned action overlaid: A1 = pre-grasp, "
            "A2 = grasp, A3 = release. Each is drawn as a 3D frame "
            "(RED = X / pinch direction, GREEN = Y / left-right, BLUE = Z / vertical)."
        )
        lines.append(
            "Planned waypoints (world frame). Format: "
            "[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad, gripper] - "
            "positions in METERS, rotations in RADIANS (scipy 'xyz' Euler), "
            "gripper in [-1=open, +1=closed]."
        )
        for key in ("A1_pregrasp", "A2_grasp", "A3_release", "A4_home"):
            pose = waypoints.get(key)
            if not pose:
                continue
            label = key.split("_", 1)[0]  # "A1", "A2", ...
            formatted = ", ".join(f"{float(v):+.3f}" for v in pose)
            lines.append(f"  {label} ({key}): [{formatted}]")
        if pixel_summary:
            in_frame = [p for p in pixel_summary if p.get("in_frame")]
            out_of_frame = [p for p in pixel_summary if not p.get("in_frame")]
            if in_frame:
                pieces = ", ".join(
                    f"{p['label']}@px({p['pixel_uv'][0]},{p['pixel_uv'][1]})"
                    for p in in_frame
                )
                lines.append(f"Overlay pixel positions (origin of each frame): {pieces}")
            if out_of_frame:
                names = ", ".join(p["label"] for p in out_of_frame)
                lines.append(
                    f"Note: {names} project outside the camera view (not visible on the overlay)."
                )
    elif waypoints:
        # Overlay disabled but we still hand the agent the numbers - useful
        # for debugging without the rendered axes.
        lines.append(
            "Planned waypoints (overlay disabled, world-frame numbers only). "
            "Format: [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad, gripper] - "
            "positions in METERS, rotations in RADIANS (scipy 'xyz' Euler), "
            "gripper in [-1=open, +1=closed]."
        )
        for key, pose in waypoints.items():
            formatted = ", ".join(f"{float(v):+.3f}" for v in pose)
            lines.append(f"  {key}: [{formatted}]")

    history: List[DialogueTurn] = state.get("dialogue_history", []) or []
    if history:
        lines.append("Conversation so far:")
        for t in history:
            speaker = "You (robot)" if t["role"] == "robot" else "Human"
            lines.append(f"  {speaker}: {t['content']}")
        lines.append(
            "Now look at the image again and ask the next clarifying question, "
            "or, if the human has already given a complete instruction, "
            "ask a brief one-line confirmation."
        )
    else:
        lines.append(
            "Look at the image (with the A1/A2/A3 overlay) and ask exactly ONE "
            "clarifying question that will let the robot resume."
        )
    return "\n".join(lines)


def visual_inquisitor_node(
    state: DialogueState,
    *,
    client: VisionLLMClient,
    verbose: bool = False,
) -> DialogueState:
    """LangGraph node: produces a clarifying question and appends it to history."""
    image: Optional[ImageInput] = None
    if state.get("image_b64"):
        image = ImageInput(
            b64_data=state["image_b64"],
            mime_type=state.get("image_mime", "image/png"),
        )
    elif state.get("image_path"):
        image = ImageInput.from_path(state["image_path"])

    user_prompt = _format_user_prompt(state)
    messages = [ChatMessage(role="user", content=user_prompt)]

    if verbose:
        print(f"[Agent2/{client.label}] -> asking clarifying question")
    question = client.chat(VISUAL_INQUISITOR_SYSTEM, messages, image=image).strip()
    if verbose:
        print(f"[Agent2/{client.label}] <- {question}")

    history = list(state.get("dialogue_history", []) or [])
    history.append(DialogueTurn(role="robot", content=question))

    return {
        "clarifying_question": question,
        "dialogue_history": history,
    }
