"""Dialogue-as-a-Sensor multi-agent system.

This package implements the user-facing slice of the architecture described
in README.md:

  - Agent 2 (Visual Inquisitor / NLG): looks at the stuck-state image plus the
    high-level task instruction and asks the human one clarifying question.
  - Simulated User: a separate VLM (different family from Agent 2) that plays
    the human role and answers the question grounded in the same image.
  - Agent 3 (Spatial-to-OSC Grounding / NLU): converts the human's clarified
    instruction into a JSON of per-waypoint action deltas (corrections to
    A1/A2/A3) plus an OSC delta vector ready for Agent 4.

The three are wired together as a small LangGraph state machine so that
Agent 1 (the dual-trigger monitor) and Agent 4 (the VLA executor) can be
plugged in later without restructuring the dialogue + grounding loop.
"""

from .state import DialogueState, DialogueTurn  # noqa: F401
from .graph import build_graph  # noqa: F401

__all__ = ["DialogueState", "DialogueTurn", "build_graph"]
