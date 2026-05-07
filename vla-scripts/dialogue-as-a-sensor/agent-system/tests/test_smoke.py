"""Offline smoke test - exercises the public surface without hitting any API.

Run with::

    python -m pytest agent-system/tests -q

Or as a script::

    python agent-system/tests/test_smoke.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODE_DIR = REPO_ROOT / "data-generation" / "my_data" / "episode_00000"


def _ensure_repo_src_on_path():
    src = REPO_ROOT / "agent-system" / "src"
    if str(src) not in os.sys.path:
        os.sys.path.insert(0, str(src))


_ensure_repo_src_on_path()


def test_imports():
    """Package imports without exploding (no API keys touched)."""
    import dialogue_sensor  # noqa: F401
    from dialogue_sensor import build_graph  # noqa: F401
    from dialogue_sensor.agents import (  # noqa: F401
        VISUAL_INQUISITOR_SYSTEM,
        SIMULATED_USER_SYSTEM,
        visual_inquisitor_node,
        simulated_user_node,
    )


def test_state_schema_minimal():
    from dialogue_sensor.state import DialogueState

    state: DialogueState = {
        "image_path": "x.png",
        "image_b64": "AA==",
        "image_mime": "image/png",
        "task_description": "Pick up the cereal box.",
        "trigger_reason": "semantic_ambiguity",
        "kinematic_state": {},
        "osc_delta": None,
        "dialogue_history": [],
        "clarifying_question": None,
        "clarified_instruction": None,
        "turn_count": 0,
        "max_turns": 1,
        "done": False,
        "metadata": {},
    }
    assert state["max_turns"] == 1


@pytest.mark.skipif(
    not EPISODE_DIR.is_dir(),
    reason="Run after `bash data-generation/run_generation.sh` to populate my_data/.",
)
def test_load_episode_state():
    from dialogue_sensor.episode_loader import load_episode_state

    state = load_episode_state(EPISODE_DIR, max_turns=1)
    assert state["task_description"]
    assert state["image_b64"]
    assert state["image_mime"] == "image/png"
    assert state["trigger_reason"] in {"semantic_ambiguity", "kinematic_collision", "unknown"}


@pytest.mark.skipif(
    not EPISODE_DIR.is_dir(),
    reason="Run after `bash data-generation/run_generation.sh` to populate my_data/.",
)
def test_dry_run_prompts_render():
    """The CLI's dry-run path should render both prompts without API keys."""
    from dialogue_sensor.episode_loader import load_episode_state
    from dialogue_sensor.agents.visual_inquisitor import _format_user_prompt as ip
    from dialogue_sensor.agents.simulated_user import _format_user_prompt as sp

    state = load_episode_state(EPISODE_DIR, max_turns=1)
    inq_prompt = ip(state)
    assert "Task:" in inq_prompt
    assert "Trigger reason:" in inq_prompt

    state_with_q = dict(state)
    state_with_q["clarifying_question"] = "Should I grasp the cereal from the side?"
    state_with_q["dialogue_history"] = [
        {"role": "robot", "content": state_with_q["clarifying_question"]}
    ]
    sim_prompt = sp(state_with_q)
    assert "Robot task:" in sim_prompt
    assert "Should I grasp the cereal" in sim_prompt


def test_graph_compiles_without_keys(monkeypatch):
    """build_graph itself shouldn't make any network calls or need keys."""
    from dialogue_sensor.graph import build_graph

    class _StubClient:
        @property
        def label(self) -> str: return "stub:none"
        def chat(self, system, messages, image=None):  # noqa: D401
            return "(stub)"

    g = build_graph(
        inquisitor_client=_StubClient(),
        user_client=_StubClient(),
        verbose=False,
    )
    assert g is not None  # langgraph compiled object


@pytest.mark.skipif(
    not EPISODE_DIR.is_dir(),
    reason="Run after `bash data-generation/run_generation.sh` to populate my_data/.",
)
def test_overlay_renders_on_legacy_episode():
    """Overlay should render even for episodes without saved K/T_wc."""
    from dialogue_sensor.episode_loader import load_episode_state
    from dialogue_sensor.overlays import render_overlay
    import base64
    import io
    from PIL import Image

    state = load_episode_state(EPISODE_DIR, max_turns=1, overlay=True)
    assert state["metadata"]["_overlay_used"] is True
    assert state["metadata"]["_overlay_used_default_camera"] is True
    pixel_summary = state["metadata"]["_overlay_waypoint_pixels"]
    assert any(p.get("in_frame") for p in pixel_summary), \
        "At least one waypoint should project in-frame"

    decoded = base64.b64decode(state["image_b64"])
    Image.open(io.BytesIO(decoded)).verify()  # valid PNG


def test_overlay_can_be_disabled():
    """--no_overlay path: image should be the raw bytes."""
    if not EPISODE_DIR.is_dir():
        pytest.skip("requires my_data/")
    from dialogue_sensor.episode_loader import load_episode_state
    state = load_episode_state(EPISODE_DIR, max_turns=1, overlay=False)
    assert state["metadata"]["_overlay_used"] is False


# ============================================================================
# Agent 3 (Spatial-to-OSC Grounding) - offline tests
# ============================================================================


def test_agent3_extract_json_plain():
    from dialogue_sensor.agents.spatial_grounding import extract_json_object
    result = extract_json_object('{"reasoning": "ok", "corrections": {}}')
    assert result["reasoning"] == "ok"
    assert result["corrections"] == {}


def test_agent3_extract_json_fenced():
    """LLMs sometimes wrap JSON in ```json ... ``` fences."""
    from dialogue_sensor.agents.spatial_grounding import extract_json_object
    text = (
        "Here you go:\n"
        "```json\n"
        '{"reasoning": "rotate A1", "corrections": {"A1": {"d_yaw": 1.5708}}}\n'
        "```\n"
        "Hope that helps."
    )
    result = extract_json_object(text)
    assert result["corrections"] == {"A1": {"d_yaw": 1.5708}}


def test_agent3_extract_json_balanced_scan():
    """Fallback path - last balanced {...} substring."""
    from dialogue_sensor.agents.spatial_grounding import extract_json_object
    text = 'prelude {"a": 1} more text {"reasoning": "x", "corrections": {}}'
    result = extract_json_object(text)
    assert result["reasoning"] == "x"


def test_agent3_extract_json_raises_when_unparseable():
    from dialogue_sensor.agents.spatial_grounding import extract_json_object
    with pytest.raises(ValueError):
        extract_json_object("absolutely no json here")


def test_agent3_validate_corrections_clips_and_drops():
    from dialogue_sensor.agents.spatial_grounding import validate_corrections
    import math
    raw = {
        "A1": {"d_yaw": 1.5708, "bogus_key": 1.0, "dx": 999.0},  # dx clipped
        "A2": {"dz": 0.03, "d_pitch": 50.0},                     # d_pitch clipped to π
        "A4": {"dy": 0.01},                                       # unknown waypoint
        "A3": "not a dict",                                       # bad shape
    }
    sanitised, warnings = validate_corrections(raw)
    assert sanitised["A1"]["d_yaw"] == 1.5708           # within ±π, unchanged
    assert sanitised["A1"]["dx"] == 0.10                # hard cap
    assert "bogus_key" not in sanitised["A1"]
    assert sanitised["A2"]["dz"] == 0.03
    assert sanitised["A2"]["d_pitch"] == math.pi        # rotation hard cap is +π
    assert "A4" not in sanitised
    assert "A3" not in sanitised
    # We expect at least: unknown key, unknown waypoint, bad shape, clipped dx, clipped d_pitch
    assert len(warnings) >= 3


def test_agent3_apply_corrections_position_and_rotation():
    """All deltas add directly to the absolute pose. Rotation deltas are
    already in RADIANS, so apply_corrections is a pure pose+delta sum."""
    from dialogue_sensor.agents.spatial_grounding import apply_corrections
    import math
    waypoints = {
        "A1_pregrasp": [0.10, 0.20, 1.20, 0.0, 0.0, 0.0, -1.0],
        "A2_grasp":    [0.10, 0.20, 0.93, 0.0, 0.0, 0.0,  1.0],
        "A3_release":  [0.00, 0.40, 0.90, 0.0, 0.0, 0.0, -1.0],
    }
    corrections = {
        "A1": {"d_yaw": math.pi / 2},        # +π/2 rad about z
        "A2": {"dy": -0.03, "dz": 0.02},     # -3 cm in y, +2 cm in z
    }
    out = apply_corrections(waypoints, corrections)
    a1 = out["A1_pregrasp"]
    a2 = out["A2_grasp"]
    a3 = out["A3_release"]
    # A1 yaw is exactly π/2 (no unit conversion).
    assert abs(a1[5] - math.pi / 2) < 1e-9
    assert a1[:3] == [0.10, 0.20, 1.20]    # position unchanged
    # A2 xy z deltas applied.
    assert abs(a2[1] - 0.17) < 1e-9         # 0.20 + (-0.03)
    assert abs(a2[2] - 0.95) < 1e-9         # 0.93 + 0.02
    assert a2[5] == 0.0                     # yaw untouched
    # A3 fully unchanged.
    assert a3 == [0.00, 0.40, 0.90, 0.0, 0.0, 0.0, -1.0]


def test_agent3_corrections_to_a2_osc_delta():
    """A2 dict -> 7-DoF delta vector in metres + radians (no unit conversion)."""
    from dialogue_sensor.agents.spatial_grounding import corrections_to_a2_osc_delta
    import math
    delta = corrections_to_a2_osc_delta({"A2": {"dx": 0.02, "d_yaw": math.pi / 2}})
    assert delta is not None
    assert delta[0] == 0.02
    assert abs(delta[5] - math.pi / 2) < 1e-9
    assert delta[6] == 0.0
    # No A2 entry -> None.
    assert corrections_to_a2_osc_delta({"A1": {"d_yaw": math.pi / 4}}) is None


def test_agent3_node_with_stub_client():
    """Run the full Agent 3 node end-to-end with a stubbed VLM."""
    if not EPISODE_DIR.is_dir():
        pytest.skip("requires my_data/")
    from dialogue_sensor.agents.spatial_grounding import spatial_grounding_node
    from dialogue_sensor.episode_loader import load_episode_state

    class _StubGroundingClient:
        @property
        def label(self) -> str: return "stub:ground"
        def chat(self, system, messages, image=None):
            return (
                '{"reasoning": "Rotate A1 by π/2 rad so the red axis '
                'is perpendicular to the cereal\'s broad face.", '
                '"corrections": {"A1": {"d_yaw": 1.5708}}}'
            )

    state = load_episode_state(EPISODE_DIR, max_turns=1, overlay=True)
    # Simulate the dialogue having concluded.
    state["dialogue_history"] = [
        {"role": "robot", "content": "Should I rotate A1 by π/2 rad to align with the box?"},
        {"role": "human", "content": "Yes, rotate A1 by π/2 rad clockwise."},
    ]
    state["clarified_instruction"] = "Yes, rotate A1 by π/2 rad clockwise."
    state["done"] = True
    state["turn_count"] = 1

    out = spatial_grounding_node(state, client=_StubGroundingClient(), verbose=False)
    assert out["corrections"] == {"A1": {"d_yaw": 1.5708}}
    assert "A1_pregrasp" in out["corrected_waypoints"]
    # No A2 correction -> no osc_delta.
    assert out["osc_delta"] is None
    assert "π/2" in (out["grounding_reasoning"] or "")


def test_full_graph_through_agent3_with_stubs():
    """build_graph wires inquisitor -> sim_user -> spatial_grounding -> END."""
    if not EPISODE_DIR.is_dir():
        pytest.skip("requires my_data/")
    from dialogue_sensor.graph import build_graph
    from dialogue_sensor.episode_loader import load_episode_state

    class _Q:
        @property
        def label(self) -> str: return "stub:q"
        def chat(self, system, messages, image=None):
            return "Should I shift A2 left 3 cm to clear the bottle?"

    class _A:
        @property
        def label(self) -> str: return "stub:a"
        def chat(self, system, messages, image=None):
            return "Yes, shift A2 about 3 cm in -y."

    class _G:
        @property
        def label(self) -> str: return "stub:g"
        def chat(self, system, messages, image=None):
            return '{"reasoning": "shift A2 by 3 cm in -y", "corrections": {"A2": {"dy": -0.03}}}'

    state = load_episode_state(EPISODE_DIR, max_turns=1, overlay=True)
    graph = build_graph(
        inquisitor_client=_Q(),
        user_client=_A(),
        grounding_client=_G(),
        verbose=False,
    )
    final = graph.invoke(state)
    assert final["turn_count"] == 1
    assert final["corrections"] == {"A2": {"dy": -0.03}}
    a2_orig = final["metadata"]["waypoints"]["A2_grasp"]
    a2_new = final["corrected_waypoints"]["A2_grasp"]
    assert abs(a2_new[1] - (a2_orig[1] - 0.03)) < 1e-9
    # OSC delta vector is non-None for A2 corrections.
    assert final["osc_delta"] is not None and abs(final["osc_delta"][1] - (-0.03)) < 1e-9


if __name__ == "__main__":
    import sys
    raise SystemExit(pytest.main([__file__, "-v"]))
