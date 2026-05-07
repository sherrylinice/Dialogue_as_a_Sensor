"""LangGraph wiring for the Agent 2 ↔ simulated-user dialogue loop and the
Agent 3 (Spatial-to-OSC Grounding) finaliser."""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph

from .agents.simulated_user import simulated_user_node
from .agents.spatial_grounding import spatial_grounding_node
from .agents.visual_inquisitor import visual_inquisitor_node
from .llm_clients import VisionLLMClient
from .state import DialogueState


def build_graph(
    *,
    inquisitor_client: VisionLLMClient,
    user_client: VisionLLMClient,
    grounding_client: Optional[VisionLLMClient] = None,
    verbose: bool = False,
):
    """Compile the dialogue + grounding state machine.

    Edges
    -----
        START                -> visual_inquisitor
        visual_inquisitor    -> simulated_user
        simulated_user       -> visual_inquisitor   (when more turns remain)
                              -> spatial_grounding  (when dialogue is done,
                                                      grounding enabled)
                              -> END                (when grounding disabled)
        spatial_grounding    -> END

    The conditional edge after the simulated user makes the graph a true
    cyclic state machine, matching README.md §"Conditional Routing (Edges)".
    Agent 3 runs at most once per dialogue: only when ``done`` is true and
    a grounding client was supplied.
    """
    g: StateGraph = StateGraph(DialogueState)

    g.add_node(
        "visual_inquisitor",
        lambda state: visual_inquisitor_node(state, client=inquisitor_client, verbose=verbose),
    )
    g.add_node(
        "simulated_user",
        lambda state: simulated_user_node(state, client=user_client, verbose=verbose),
    )

    g.add_edge(START, "visual_inquisitor")
    g.add_edge("visual_inquisitor", "simulated_user")

    if grounding_client is not None:
        g.add_node(
            "spatial_grounding",
            lambda state: spatial_grounding_node(state, client=grounding_client, verbose=verbose),
        )

    def _should_continue(state: DialogueState) -> str:
        """Route after the simulated user.

        - Loop back to the inquisitor while turns remain.
        - When the dialogue is done, route to Agent 3 if it was supplied,
          otherwise END.
        """
        max_turns = int(state.get("max_turns", 1))
        turn_count = int(state.get("turn_count", 0))
        if not state.get("done", False) and turn_count < max_turns:
            return "loop"
        if grounding_client is not None:
            return "ground"
        return "end"

    g.add_conditional_edges(
        "simulated_user",
        _should_continue,
        {
            "loop": "visual_inquisitor",
            "ground": "spatial_grounding" if grounding_client is not None else END,
            "end": END,
        },
    )

    if grounding_client is not None:
        g.add_edge("spatial_grounding", END)

    return g.compile()
