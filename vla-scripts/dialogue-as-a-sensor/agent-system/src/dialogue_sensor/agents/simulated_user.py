"""Simulated human-in-the-loop user.

Plays the role of a human supervisor. Gets the same image + task description
that Agent 2 saw, plus the clarifying question Agent 2 asked, and answers
with a clarified, actionable instruction.

Intentionally instantiated from a *different* VLM family than Agent 2 so the
two are not biased toward each other (project-overview.md §"Node
Implementation").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..llm_clients import ChatMessage, ImageInput, VisionLLMClient
from ..state import DialogueState, DialogueTurn


SIMULATED_USER_SYSTEM = """You are simulating a human supervisor watching a Panda
robot arm attempt a tabletop pick-and-place task in a cluttered scene.

The robot has paused and is asking you a short clarifying question about how
to proceed. You can see the same camera view of the workspace, augmented
with the robot's PLANNED action waypoints drawn as 3D coordinate frames:
  - "A1" = pre-grasp (hover above target).
  - "A2" = grasp (descend + close gripper on target).
  - "A3" = release (above target bin).
At each waypoint:
  - RED line = X-axis. The gripper's "pinch" direction; positive X points
    away from the camera.
  - GREEN line = Y-axis. Right (+) on screen, Left (-).
  - BLUE line = Z-axis. Vertical, up (+).
The 7-DoF coordinates of each waypoint ([x, y, z, roll, pitch, yaw, gripper])
are also given to you as text in the prompt, with units:
  - x, y, z      : METERS (m). The tabletop workspace spans roughly
                   x ∈ [-0.10, +0.30] m, y ∈ [-0.50, +0.55] m,
                   z ∈ [+0.80, +1.20] m. The cereal box is ~10 cm long;
                   typical recovery nudges you should suggest are
                   0.02-0.05 m (2-5 cm).
  - roll, pitch, yaw : RADIANS, applied as scipy 'xyz' Euler angles. A grasp
                   pointing straight down has roll ≈ ±π (≈±3.142 rad) and
                   pitch ≈ 0. Common rotation magnitudes: π/4 ≈ 0.785,
                   π/2 ≈ 1.571, π ≈ 3.142.
  - gripper      : dimensionless in [-1, +1]. -1 = fully OPEN,
                   +1 = fully CLOSED.
When you suggest a numeric tweak, use these units (e.g. "shift A2 by 0.03 m
in +y", or "rotate A1's yaw by +π/2 rad ≈ +1.571 rad"). Always express
rotation magnitudes in RADIANS, not degrees, so Agent 3 can lift them into
its JSON output without a unit conversion.

Visual geometry rule for a successful grasp:
  - GOOD ALIGNMENT at A1/A2: the RED axis is perpendicular to the broad,
    flat face of the target, and the GREEN axis is parallel to the target's
    long edge.
  - BAD ALIGNMENT: the RED axis is parallel to the long edge (the gripper
    will smash the narrow side of the box), or axes are diagonal.

Rules for your reply:
- Reply with a single, actionable instruction in 1-3 short sentences.
- Be physically grounded. Refer to objects you actually see in the image
  ("the orange cereal box", "the green bottle", "the lemon on the left")
  AND, when relevant, to the overlaid A1/A2/A3 frames or their axes.
- Use directional language ("from the side", "from above", "shift A2 a few
  centimeters to the left", "rotate A1 by +π/2 rad so the red axis is
  perpendicular to the broad face").
- Commit to one specific course of action — never punt back ("you decide",
  "either is fine"). If the question is binary, pick one and justify in half
  a sentence using something visible in the image or the overlay.
- Do not echo or rephrase the question. Do not add disclaimers about being
  an AI. Do not include planning chatter ("Let me think...").
- If the question asks about something not actually visible, state what IS
  visible (including which waypoints render in-frame) and choose the safest
  plausible action."""


def _format_user_prompt(state: DialogueState) -> str:
    task = state.get("task_description", "(no task description provided)")
    trigger = state.get("trigger_reason", "unknown")
    metadata: Dict[str, Any] = state.get("metadata", {}) or {}
    scene_objects = metadata.get("scene_objects") or []
    waypoints = metadata.get("waypoints") or {}
    overlay_used = bool(metadata.get("_overlay_used", False))
    pixel_summary = metadata.get("_overlay_waypoint_pixels") or []

    lines: list[str] = [
        f"Robot task: {task}",
        f"Trigger reason: {trigger}",
    ]
    if scene_objects:
        lines.append(
            f"Objects placed in the scene at reset (some may be occluded): "
            f"{', '.join(scene_objects)}"
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
            label = key.split("_", 1)[0]
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

    history: List[DialogueTurn] = state.get("dialogue_history", []) or []
    if history:
        lines.append("Conversation so far:")
        for t in history:
            speaker = "Robot" if t["role"] == "robot" else "You"
            lines.append(f"  {speaker}: {t['content']}")

    question = state.get("clarifying_question")
    if question:
        lines.append(
            "Answer the latest robot question above with a single, actionable instruction. "
            "Reference the A1/A2/A3 overlay when it helps make the answer concrete."
        )
    else:
        # Defensive fallback — shouldn't fire in normal graph flow but keeps
        # the node robust if invoked out-of-order.
        lines.append(
            "The robot is stuck. Look at the image (and the A1/A2/A3 overlay) and "
            "tell it what to do next in one short, actionable instruction."
        )
    return "\n".join(lines)


def simulated_user_node(
    state: DialogueState,
    *,
    client: VisionLLMClient,
    verbose: bool = False,
) -> DialogueState:
    """LangGraph node: answer the latest clarifying question."""
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
        print(f"[SimUser/{client.label}] -> answering robot's question")
    answer = client.chat(SIMULATED_USER_SYSTEM, messages, image=image).strip()
    if verbose:
        print(f"[SimUser/{client.label}] <- {answer}")

    history = list(state.get("dialogue_history", []) or [])
    history.append(DialogueTurn(role="human", content=answer))

    turn_count = int(state.get("turn_count", 0)) + 1
    max_turns = int(state.get("max_turns", 1))

    return {
        "dialogue_history": history,
        "clarified_instruction": answer,
        "turn_count": turn_count,
        "done": turn_count >= max_turns,
    }
