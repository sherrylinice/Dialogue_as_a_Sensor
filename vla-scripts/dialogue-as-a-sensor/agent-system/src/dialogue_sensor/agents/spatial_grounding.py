"""Agent 3 — the Spatial-to-OSC Grounding agent (NLU node).

Reads the human-robot conversation that just resolved an uncertainty, looks
at the workspace image (with the planned waypoints overlaid), and emits a
JSON object of action deltas that correct the planned waypoints A1/A2/A3
to enact the human's clarified instruction.

The output JSON conforms to the convention already used by
``data-generation/action_correction_prompt.py``:

    {
        "reasoning": "<chain of thought>",
        "corrections": {
            "A1": {"d_yaw": 90.0},
            "A2": {"dx": 0.02, "dz": -0.01}
        }
    }

Conventions
-----------
* ``dx, dy, dz``: METERS. Sign convention matches the visualised axes —
  +x away from camera (red axis), +y toward viewer's right (green axis),
  +z up (blue axis).
* ``d_roll, d_pitch, d_yaw``: RADIANS, scipy ``'xyz'`` Euler convention.
  Same units as the absolute waypoint angles in ``metadata.waypoints``,
  so ``corrected_waypoints`` is just ``waypoint + delta`` in every
  channel — no unit conversion happens anywhere in the pipeline.
  Common values: ±π/4 ≈ ±0.785, ±π/2 ≈ ±1.571, ±π ≈ ±3.142.
* ``d_gripper``: dimensionless in [-2, +2] (since the absolute gripper
  is in [-1, +1]).

If a waypoint requires no correction the key is omitted from
``corrections``. If NO waypoint requires correction at all,
``corrections`` is ``{}`` and the original waypoints pass through
unchanged.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from ..llm_clients import ChatMessage, ImageInput, VisionLLMClient
from ..state import DialogueState, DialogueTurn


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


SPATIAL_GROUNDING_SYSTEM = """You are the spatial to action-grouding agent 
inside a robotic-manipulation pipeline.

A Panda robot arm executing a tabletop pick-and-place task in Robosuite hit
an uncertainty trigger. The robot has asked a clarifying question to a human 
user, and the user gave a clarified instruction. 
Your job: convert that instruction into precise, machine-actionable action
deltas that correct the planned waypoints A1/A2/A3 (corresponding to action 1, 
action 2, and action 3 of the key waypoints in an action plan to perform a task).

You will receive in the user prompt:
- An image of the state with the planned waypoints overlaid as 3D coordinate
  frames (RED = X / pinch, GREEN = Y / left-right, BLUE = Z / vertical;
  axis length 5 cm).
- The current 7-DoF waypoints in the world frame, format
  [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad, gripper], with units:
  positions in METERS, rotations in RADIANS (scipy 'xyz' Euler), gripper
  in [-1=open, +1=closed].
- The trigger_reason (semantic_ambiguity or kinematic_collision).
- The full robot-human dialogue, ending with the human's clarified
  instruction.

Output exactly one JSON object with this schema (and nothing else):

  {
    "reasoning": "<concise chain of thought, 1-3 sentences>",
    "corrections": {
      "A1": {"<delta-key>": <number>, ...},
      "A2": {"<delta-key>": <number>, ...},
      "A3": {"<delta-key>": <number>, ...}
    }
  }

Allowed delta keys per waypoint:
  - "dx", "dy", "dz"               -> position deltas in METERS
                                        +x away from camera (red axis),
                                        +y to viewer's right (green axis),
                                        +z up (blue axis).
  - "d_roll", "d_pitch", "d_yaw"   -> rotation deltas in RADIANS (scipy 'xyz'
                                        Euler), SAME units as the absolute
                                        waypoint angles. Common values:
                                        ±π/4 ≈ ±0.785, ±π/2 ≈ ±1.571,
                                        ±π ≈ ±3.142.
  - "d_gripper"                    -> dimensionless gripper delta in [-2, +2]
                                        (since absolute gripper is in [-1, +1]).

Rules:
- Output ONLY the JSON object. No code fences, no preamble, no trailing prose.
- Omit delta keys that don't change. Omit a waypoint key entirely if that
  waypoint is unchanged. If NO correction is needed, output
  {"reasoning": "...", "corrections": {}}.
- Magnitude guidelines:
    * Position deltas: typically 0.02-0.05 m (2-5 cm).
    * Rotation deltas: typically clean multiples of π/4 (≈0.785 rad) or
      pi/2 (≈1.571 rad).
    * Gripper deltas: rare; usually leave gripper as-is, only fix if needed.
- If the human's instruction is a SCENE manipulation that the existing
  waypoints already accomplish (e.g. "the cereal is the orange box A2 is
  on, proceed"), output empty corrections {}.
- If the human asked the robot to MOVE / PUSH / REROUTE around an obstacle,
  shift the relevant waypoint position(s) in the indicated
  direction (or, when a re-orient is needed, apply a clean d_yaw rotation
  of ±π/2 radians).
- Cross-check with the visual geometry rule for grasp success:
    GOOD ALIGNMENT: at A1/A2 the RED axis is perpendicular to the broad
                   flat face of the target; GREEN axis parallel to long edge.
    BAD ALIGNMENT: RED parallel to long edge -> apply d_yaw of ±π/2
                   (≈±1.571 rad) to A1.
- If the human speaks in DEGREES (e.g. "rotate 90 degree"), convert mentally to
  RADIANS (90 degree = pi/2 ≈ 1.5708 rad) before emitting the JSON. The JSON
  output is ALWAYS in radians.

Decision protocol (follow internally, do NOT print):
  1. Read the human's last message. What action did they tell the robot?
  2. Identify which waypoint(s) the action affects (A1 / A2 / A3 / none).
  3. Choose the smallest set of delta keys that produces the action.
  4. Cross-check against the visual geometry rule.
  5. Emit the JSON envelope.

Few-shot examples follow."""


# Built-in few-shot examples. Mirror the patterns from
# ``data-generation/action_correction_prompt.py`` but adapt them to the
# JSON-envelope format Agent 3 must emit.
SPATIAL_GROUNDING_EXAMPLES: List[Tuple[str, Dict[str, Any]]] = [
    (
        "Trigger: kinematic_collision. Human: 'Rotate A1 by π/2 radians "
        "(about 1.571 rad) so the red axis is perpendicular to the broad "
        "face of the cereal.'",
        {
            "reasoning": (
                "Human asked for a +pi/2 rad yaw rotation at A1 to fix grasp "
                "alignment. Apply d_yaw=+1.5708 to A1 only; A2/A3 unchanged."
            ),
            "corrections": {"A1": {"d_yaw": 1.5708}},
        },
    ),
    (
        "Trigger: kinematic_collision. Human: 'Rotate A1 90° so the red axis "
        "is perpendicular to the broad face of the cereal.'",
        {
            "reasoning": (
                "Human spoke in degrees (90 degree) but the JSON convention is "
                "radians; 90 degree = pi/2 ≈ 1.5708 rad. Apply d_yaw=+1.5708 to "
                "A1 only."
            ),
            "corrections": {"A1": {"d_yaw": 1.5708}},
        },
    ),
    (
        "Trigger: kinematic_collision. Human: 'Shift A2 about 3 cm to the "
        "left in y to clear the green bottle.'",
        {
            "reasoning": (
                "Human wants A2 nudged 3 cm in -y. Apply dy=-0.03 to A2 only."
            ),
            "corrections": {"A2": {"dy": -0.03}},
        },
    ),
    (
        "Trigger: semantic_ambiguity. Human: 'Yes, A2 is centered on the "
        "cereal box and the red axis is correctly aligned. Proceed.'",
        {
            "reasoning": (
                "Confirmation - no waypoint correction needed; the planned "
                "A1/A2/A3 are already correct."
            ),
            "corrections": {},
        },
    ),
    (
        "Trigger: kinematic_collision. Human: 'A2 is dipping too deep into "
        "the box; lift it ~2 cm.'",
        {
            "reasoning": (
                "Raise A2 by 2 cm to avoid clipping into the cereal. "
                "Apply dz=+0.02 to A2."
            ),
            "corrections": {"A2": {"dz": 0.02}},
        },
    ),
    (
        "Trigger: semantic_ambiguity. Human: 'The cereal is the brown box at "
        "front-right of the tray, not the green bottle that A2 is on. Move "
        "A2 to the brown box, about 4 cm in +y and 2 cm in -x.'",
        {
            "reasoning": (
                "Human relocated the grasp target. Shift A2 +0.04 m in y and "
                "-0.02 m in x. A1 sits above A2 in the same xy, so apply the "
                "same xy nudge to A1 to keep the descent vertical."
            ),
            "corrections": {
                "A1": {"dy": 0.04, "dx": -0.02},
                "A2": {"dy": 0.04, "dx": -0.02},
            },
        },
    ),
]


# ---------------------------------------------------------------------------
# Constants for validation
# ---------------------------------------------------------------------------

POSITION_DELTA_KEYS = ("dx", "dy", "dz")
ROTATION_DELTA_KEYS = ("d_roll", "d_pitch", "d_yaw")
GRIPPER_DELTA_KEYS = ("d_gripper",)
ALL_DELTA_KEYS = POSITION_DELTA_KEYS + ROTATION_DELTA_KEYS + GRIPPER_DELTA_KEYS

# Hard caps - anything beyond these is clipped (with a warning) since LLMs
# occasionally produce wild numbers like dy=10 (10 m off the table).
MAX_POSITION_DELTA_M = 0.10
MAX_ROTATION_DELTA_RAD = math.pi   # ±π rad ≈ ±180°; full rotation makes
                                   # no sense as a *correction*.
MAX_GRIPPER_DELTA = 2.0

# Maps short waypoint label ("A1") to canonical waypoint key ("A1_pregrasp")
# and back. Agent 3 emits short labels; the rest of the system speaks long
# names.
LABEL_TO_KEY = {
    "A1": "A1_pregrasp",
    "A2": "A2_grasp",
    "A3": "A3_release",
}
KEY_TO_LABEL = {v: k for k, v in LABEL_TO_KEY.items()}


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _format_examples_block() -> str:
    blocks = []
    for i, (user_text, response) in enumerate(SPATIAL_GROUNDING_EXAMPLES, start=1):
        blocks.append(
            f"Example {i}:\nUser: {user_text}\nAssistant: {json.dumps(response)}"
        )
    return "\n\n".join(blocks)


def _format_user_prompt(state: DialogueState) -> str:
    task = state.get("task_description", "(no task description provided)")
    trigger = state.get("trigger_reason", "unknown")
    metadata: Dict[str, Any] = state.get("metadata", {}) or {}
    waypoints: Dict[str, List[float]] = metadata.get("waypoints") or {}
    overlay_used = bool(metadata.get("_overlay_used", False))
    pixel_summary = metadata.get("_overlay_waypoint_pixels") or []
    scene_objects = metadata.get("scene_objects") or []
    stuck_phase = metadata.get("stuck_phase")

    lines: list[str] = [
        f"Task: {task}",
        f"Trigger reason: {trigger}",
    ]
    if stuck_phase:
        lines.append(f"Trajectory phase where the robot stalled: {stuck_phase}")
    if scene_objects:
        lines.append(
            f"Objects placed in the scene at reset (some may be occluded): "
            f"{', '.join(scene_objects)}"
        )

    if overlay_used:
        lines.append(
            "The image has the robot's planned action overlaid: A1 = pre-grasp, "
            "A2 = grasp, A3 = release. RED = X / pinch direction, GREEN = Y / "
            "left-right, BLUE = Z / vertical."
        )
    if waypoints:
        lines.append(
            "Current planned waypoints (world frame). Format: "
            "[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad, gripper] - positions "
            "in METERS, rotations in RADIANS, gripper in [-1=open, +1=closed]."
        )
        for key in ("A1_pregrasp", "A2_grasp", "A3_release", "A4_home"):
            pose = waypoints.get(key)
            if not pose:
                continue
            label = KEY_TO_LABEL.get(key, key.split("_", 1)[0])
            formatted = ", ".join(f"{float(v):+.3f}" for v in pose)
            lines.append(f"  {label} ({key}): [{formatted}]")
        if pixel_summary:
            in_frame = [p for p in pixel_summary if p.get("in_frame")]
            if in_frame:
                pieces = ", ".join(
                    f"{p['label']}@px({p['pixel_uv'][0]},{p['pixel_uv'][1]})"
                    for p in in_frame
                )
                lines.append(f"Overlay pixel positions (origin of each frame): {pieces}")

    history: List[DialogueTurn] = state.get("dialogue_history", []) or []
    if history:
        lines.append("Robot ↔ human dialogue (most recent last):")
        for t in history:
            speaker = "Robot" if t["role"] == "robot" else "Human"
            lines.append(f"  {speaker}: {t['content']}")
    clarified = state.get("clarified_instruction")
    if clarified:
        lines.append(f"Final clarified instruction from the human: {clarified}")

    lines.append(
        "Now emit the JSON object {\"reasoning\": ..., \"corrections\": {...}} "
        "describing the smallest set of action deltas that enact the human's "
        "instruction. Use the units and conventions from the system prompt."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Robustly pull the JSON envelope out of an LLM response.

    Tries (in order): direct ``json.loads``, the last fenced ``json`` code
    block, the last balanced ``{...}`` substring. Raises ``ValueError`` if
    nothing parses.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response from spatial grounding agent.")

    # 1. Plain JSON.
    try:
        return _ensure_dict(json.loads(text))
    except json.JSONDecodeError:
        pass

    # 2. Fenced code blocks. Take the LAST one (some models emit a
    #    "scratchpad" example before the real answer).
    fenced = _FENCED_JSON_RE.findall(text)
    for candidate in reversed(fenced):
        try:
            return _ensure_dict(json.loads(candidate))
        except json.JSONDecodeError:
            continue

    # 3. Balanced-brace scan. Find the LAST top-level "{...}" and try to parse.
    last_obj = _find_last_balanced_object(text)
    if last_obj is not None:
        try:
            return _ensure_dict(json.loads(last_obj))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from spatial-grounding output. "
        f"First 200 chars: {text[:200]!r}"
    )


def _ensure_dict(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object at top level, got {type(obj).__name__}")
    return obj


def _find_last_balanced_object(text: str) -> Optional[str]:
    """Scan ``text`` for the last balanced ``{...}`` substring."""
    last: Optional[str] = None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    last = text[start : i + 1]
                    start = -1
    return last


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_corrections(raw: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Sanitise a raw ``corrections`` dict into ``{label: {key: float}}``.

    Returns the sanitised dict and a list of warnings (clipped values,
    dropped keys). Unknown waypoint labels and unknown delta keys are
    dropped silently except for being recorded in the warnings list.
    """
    warnings: List[str] = []
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(raw, dict):
        warnings.append(f"corrections is not an object ({type(raw).__name__}); ignoring.")
        return out, warnings

    for label, deltas in raw.items():
        if label not in LABEL_TO_KEY:
            warnings.append(f"Unknown waypoint label '{label}' (allowed: A1, A2, A3); skipping.")
            continue
        if not isinstance(deltas, dict):
            warnings.append(f"corrections['{label}'] is not an object; skipping.")
            continue
        sanitised: Dict[str, float] = {}
        for key, val in deltas.items():
            if key not in ALL_DELTA_KEYS:
                warnings.append(
                    f"corrections['{label}']: unknown delta key '{key}'; dropping."
                )
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                warnings.append(
                    f"corrections['{label}']['{key}']: non-numeric value {val!r}; dropping."
                )
                continue
            if not math.isfinite(fval):
                warnings.append(
                    f"corrections['{label}']['{key}']: non-finite value {fval}; dropping."
                )
                continue
            clipped = _clip_delta(key, fval)
            if clipped != fval:
                warnings.append(
                    f"corrections['{label}']['{key}']: clipped {fval:+.4f} -> {clipped:+.4f}."
                )
            sanitised[key] = clipped
        if sanitised:
            out[label] = sanitised

    return out, warnings


def _clip_delta(key: str, value: float) -> float:
    if key in POSITION_DELTA_KEYS:
        return max(-MAX_POSITION_DELTA_M, min(MAX_POSITION_DELTA_M, value))
    if key in ROTATION_DELTA_KEYS:
        return max(-MAX_ROTATION_DELTA_RAD, min(MAX_ROTATION_DELTA_RAD, value))
    if key in GRIPPER_DELTA_KEYS:
        return max(-MAX_GRIPPER_DELTA, min(MAX_GRIPPER_DELTA, value))
    return value


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_corrections(
    waypoints: Dict[str, List[float]],
    corrections: Dict[str, Dict[str, float]],
) -> Dict[str, List[float]]:
    """Apply per-waypoint deltas to the absolute waypoints.

    Position deltas (m) are added to ``[x, y, z]`` directly. Rotation
    deltas (rad) are added directly to ``[roll, pitch, yaw]`` (no unit
    conversion - both the deltas and the absolute angles are radians).
    Gripper delta is added then clamped to [-1, +1].

    Waypoints not mentioned in ``corrections`` pass through unchanged. The
    returned dict always covers ``A1_pregrasp / A2_grasp / A3_release`` if
    they exist in ``waypoints`` (so downstream consumers can iterate
    without checking key membership).
    """
    out: Dict[str, List[float]] = {}
    for key, pose in waypoints.items():
        out[key] = [float(v) for v in pose]

    for label, deltas in corrections.items():
        wp_key = LABEL_TO_KEY.get(label)
        if wp_key is None or wp_key not in out:
            continue
        pose = out[wp_key]
        if "dx" in deltas:
            pose[0] += deltas["dx"]
        if "dy" in deltas:
            pose[1] += deltas["dy"]
        if "dz" in deltas:
            pose[2] += deltas["dz"]
        if "d_roll" in deltas:
            pose[3] += deltas["d_roll"]
        if "d_pitch" in deltas:
            pose[4] += deltas["d_pitch"]
        if "d_yaw" in deltas:
            pose[5] += deltas["d_yaw"]
        if "d_gripper" in deltas:
            pose[6] = max(-1.0, min(1.0, pose[6] + deltas["d_gripper"]))
        out[wp_key] = pose
    return out


def corrections_to_a2_osc_delta(
    corrections: Dict[str, Dict[str, float]],
) -> Optional[List[float]]:
    """Convert the A2 corrections dict to a 7-DoF OSC delta vector.

    Returns ``None`` if A2 is not in ``corrections``. The returned vector
    is ``[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]`` in
    meters + radians + dimensionless, matching the units of the absolute
    waypoint pose. Rotation deltas pass through unchanged (already radians).
    """
    a2 = corrections.get("A2")
    if not a2:
        return None
    return [
        float(a2.get("dx", 0.0)),
        float(a2.get("dy", 0.0)),
        float(a2.get("dz", 0.0)),
        float(a2.get("d_roll", 0.0)),
        float(a2.get("d_pitch", 0.0)),
        float(a2.get("d_yaw", 0.0)),
        float(a2.get("d_gripper", 0.0)),
    ]


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


def _build_messages(state: DialogueState) -> List[ChatMessage]:
    messages: List[ChatMessage] = []
    # In-context examples as alternating user/assistant turns. Image is
    # attached to the LAST user message (i.e. the live one), per the
    # contract of OpenAIVisionClient / GoogleVisionClient.
    for example_user, example_response in SPATIAL_GROUNDING_EXAMPLES:
        messages.append(ChatMessage(role="user", content=example_user))
        messages.append(
            ChatMessage(role="assistant", content=json.dumps(example_response))
        )
    messages.append(ChatMessage(role="user", content=_format_user_prompt(state)))
    return messages


def spatial_grounding_node(
    state: DialogueState,
    *,
    client: VisionLLMClient,
    verbose: bool = False,
) -> DialogueState:
    """LangGraph node: parse the conversation -> JSON deltas -> corrected waypoints."""
    image: Optional[ImageInput] = None
    if state.get("image_b64"):
        image = ImageInput(
            b64_data=state["image_b64"],
            mime_type=state.get("image_mime", "image/png"),
        )
    elif state.get("image_path"):
        image = ImageInput.from_path(state["image_path"])

    messages = _build_messages(state)
    if verbose:
        print(f"[Agent3/{client.label}] -> grounding clarified instruction to OSC deltas")

    raw_response = client.chat(SPATIAL_GROUNDING_SYSTEM, messages, image=image)
    if verbose:
        print(f"[Agent3/{client.label}] <- {raw_response.strip()[:300]}")

    try:
        envelope = extract_json_object(raw_response)
    except ValueError as e:
        if verbose:
            print(f"[Agent3/{client.label}] WARN: could not parse JSON ({e}); falling back to {{}}")
        envelope = {"reasoning": f"(JSON parse failed: {e})", "corrections": {}}

    raw_corrections = envelope.get("corrections", {}) or {}
    sanitised, warnings = validate_corrections(raw_corrections)
    if verbose:
        for w in warnings:
            print(f"[Agent3/{client.label}] WARN: {w}")

    metadata: Dict[str, Any] = state.get("metadata", {}) or {}
    original_waypoints: Dict[str, List[float]] = metadata.get("waypoints") or {}
    corrected = apply_corrections(original_waypoints, sanitised)
    osc_delta = corrections_to_a2_osc_delta(sanitised)

    reasoning_text = envelope.get("reasoning") or None
    if reasoning_text is not None:
        reasoning_text = str(reasoning_text).strip() or None

    return {
        "corrections": sanitised,
        "corrected_waypoints": corrected,
        "grounding_reasoning": reasoning_text,
        "osc_delta": osc_delta,
    }
