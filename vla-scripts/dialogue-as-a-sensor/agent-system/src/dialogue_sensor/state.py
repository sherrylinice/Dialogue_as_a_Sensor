"""LangGraph state schema shared across the dialogue nodes."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


TriggerReason = Literal["semantic_ambiguity", "kinematic_collision", "unknown"]


class DialogueTurn(TypedDict):
    """A single turn in the human ↔ robot conversation."""
    role: Literal["robot", "human"]
    content: str


class DialogueState(TypedDict, total=False):
    """LangGraph state for the dialogue loop.

    Mirrors (a subset of) the shared state from README.md §"The LangGraph
    State Definition" — the fields used by Agents 1, 2, 3 and the simulated
    user. Agent 4 (VLA executor) is out of scope for this package; the
    ``osc_delta`` slot is preserved so an Agent 4 implementation can read it.
    """

    # === Inputs (set by the caller, not mutated by nodes) ===
    image_path: str
    image_b64: str
    image_mime: str
    task_description: str
    trigger_reason: TriggerReason

    # === Forward-compatibility (Agent 1) ===
    # Populated by the kinematic-collision trigger when available.
    kinematic_state: Dict[str, Any]

    # === Dialogue rolling state (mutated by Agents 2 / sim-user) ===
    dialogue_history: List[DialogueTurn]
    clarifying_question: Optional[str]
    clarified_instruction: Optional[str]

    # === Loop control ===
    turn_count: int
    max_turns: int
    done: bool

    # === Agent 3 (Spatial-to-OSC Grounding) outputs ===
    # Per-waypoint delta dict: e.g. {"A1": {"d_yaw": 1.5708}, "A2": {"dy": -0.02}}.
    # Position keys (dx, dy, dz) are in METERS; rotation keys (d_roll, d_pitch,
    # d_yaw) are in RADIANS (same units as the absolute waypoint angles);
    # d_gripper is dimensionless. Empty {} if no correction is needed.
    corrections: Dict[str, Dict[str, float]]
    # Result of applying ``corrections`` to the original waypoints. Keys
    # match the original waypoint names ("A1_pregrasp", "A2_grasp",
    # "A3_release"); values are 7-DoF lists [x_m, y_m, z_m, roll_rad,
    # pitch_rad, yaw_rad, gripper].
    corrected_waypoints: Dict[str, List[float]]
    # Agent 3's chain-of-thought, captured from the JSON envelope for
    # debugging / dataset construction.
    grounding_reasoning: Optional[str]
    # Convenience: the "primary" 7-DoF delta sent to the executor (Agent 4).
    # Default convention: the A2_grasp delta as a 7-vector
    # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper] in METERS + RADIANS
    # (same units as the absolute waypoint poses). None if no A2 correction.
    osc_delta: Optional[List[float]]

    # === Free-form context passed in by the caller (e.g. metadata.json) ===
    metadata: Dict[str, Any]
